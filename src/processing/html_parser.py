"""
html_parser.py
~~~~~~~~~~~~~~
HTML → Page (a tree of Sections with breadcrumbs).

Site-profile path (imi.pmf.kg.ac.rs)
------------------------------------
Program pages keep every program version in Foundation tab panels:

    ul.tabs > li.tab-title > a#trigger[href="#panelX"]      (labels)
    div.tabs-content > div.content#panelX                    (panels, nested)

Each panel becomes its own Section whose breadcrumb is built from the
DOM nesting + the tab label of the ul.tabs that belongs to the panel's
own tabs-content container.  Panel IDs are NEVER interpreted — they are
copy-pasted between pages (e.g. #panel2019rac_nauke is "Teorijska
matematika…" on the math page).

Generic path
------------
trafilatura (markdown, tables kept) → sections split at headings;
BeautifulSoup body text as the last resort.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from src.processing.fetcher import canonicalize_url, display_url, is_document_url
from src.processing.schemas import Block, Page, Section
from src.processing.site_profiles import SiteProfile, profile_for
from src.processing.tables import render_table
from src.utils.text import normalize_text, slugify

logger = logging.getLogger(__name__)

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_TAGS = {
    "script", "style", "noscript", "iframe", "form", "button", "svg", "select",
    "input", "textarea", "template", "canvas", "video", "audio", "map", "object",
    "embed", "img", "picture", "head", "link", "meta",
}
BLOCK_TAGS = {
    "p", "div", "section", "article", "ul", "ol", "dl", "dt", "dd", "blockquote",
    "pre", "hr", "center", "header", "footer", "main", "aside", "figure",
    "figcaption", "address", "tr", "td", "th", "caption", "nav",
}


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _classes(tag: Tag) -> list[str]:
    return tag.get("class") or []


def _is_tabs_ul(tag) -> bool:
    return isinstance(tag, Tag) and tag.name == "ul" and "tabs" in _classes(tag)


def _is_panel(tag) -> bool:
    """Foundation tab panel: div.content#id directly inside div.tabs-content."""
    return (
        isinstance(tag, Tag)
        and tag.name == "div"
        and "content" in _classes(tag)
        and bool(tag.get("id"))
        and isinstance(tag.parent, Tag)
        and "tabs-content" in _classes(tag.parent)
    )


def _style_hidden(tag: Optional[Tag]) -> bool:
    if tag is None:
        return False
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _collect_links(root: Tag, base_url: str) -> tuple[list[str], list[str]]:
    pages: list[str] = []
    docs: list[str] = []
    for a in root.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        if urlsplit(absolute).scheme not in ("http", "https"):
            continue
        target = docs if is_document_url(absolute) else pages
        if absolute not in target:
            target.append(absolute)
    return pages, docs


@dataclass
class _PanelInfo:
    panel_id: str
    label: str
    trigger: Optional[str]
    active: bool
    ui_hidden: bool
    tab_hidden: bool                        # li.tab-title.hide — opened via JS links


_ONCLICK_TRIGGER_RE = re.compile(r"#([A-Za-z0-9_\-]+)['\"]?\s*\)\s*\.click")


def _own_text(tag: Tag) -> str:
    """Direct text of a tag (not of its element children)."""
    return normalize_text(" ".join(s for s in tag.find_all(string=True, recursive=False)))


def _has_block_children(tag: Tag) -> bool:
    return tag.find(["div", "p", "table", "ul", "ol", "section", "article"]) is not None


# --------------------------------------------------------------------------- #
#  Site-profile parser                                                         #
# --------------------------------------------------------------------------- #

class _Walker:
    """Walks the content root and emits Sections + Blocks in DOM order."""

    def __init__(
        self,
        profile: SiteProfile,
        title: str,
        panel_info: dict[int, _PanelInfo],
        top_triggers: dict[str, str],
        nav_parents: dict[str, str],
    ):
        self.profile = profile
        self.title = title
        self.panel_info = panel_info
        self.top_triggers = top_triggers
        self.nav_parents = nav_parents
        self.sections: list[Section] = []
        self._buffers: dict[int, list[str]] = {}
        self._headings: dict[int, list[tuple[int, str]]] = {}
        self._id_counter: Counter = Counter()
        self._own_top: dict[str, str] = {}
        self._info = {info.panel_id: info for info in panel_info.values()}
        self._noise = [re.compile(p, re.I) for p in profile.noise_patterns]

    # ------------------------------------------------------------------ #

    def run(self, root: Tag) -> list[Section]:
        main = Section(
            section_id="main", dom_path="main", label=self.title,
            breadcrumb=[self.title],
        )
        self._register(main)
        self._walk(root, main)
        self._flush(main)
        self._apply_click_paths()
        return self.sections

    def _apply_click_paths(self) -> None:
        """
        Rebuild breadcrumb / top tab / reachability along the visitor's
        click path.  A panel is reachable if its (click-path) parent is
        reachable and it is the default tab, has a visible tab title, or
        is opened by a JS link inside a reachable panel.  Unreachable
        panels (orphans, copy-pasted leftovers — e.g. the informatics 2018
        program hidden inside the math page) are invisible to visitors.
        """
        by_id = {s.section_id: s for s in self.sections}
        by_panel: dict[str, Section] = {}
        for s in self.sections:
            if s.panel_chain:
                by_panel.setdefault(s.panel_chain[-1], s)

        resolved: dict[str, tuple[list[str], Optional[str], bool]] = {}

        def resolve(section: Section, depth: int = 0) -> tuple[list[str], Optional[str], bool]:
            if section.section_id in resolved:
                return resolved[section.section_id]
            if section.parent_id is None or depth > 25:
                result = (section.breadcrumb, section.top_trigger, section.parent_id is None)
            else:
                own_panel = section.panel_chain[-1] if section.panel_chain else ""
                info = self._info.get(own_panel)
                via_link = None
                if info is not None and info.tab_hidden:
                    candidate = by_panel.get(self.nav_parents.get(own_panel, ""))
                    if candidate is not None and own_panel not in candidate.panel_chain:
                        via_link = candidate
                parent = via_link or by_id[section.parent_id]
                parent_crumb, parent_top, parent_reachable = resolve(parent, depth + 1)
                opens = info is None or info.active or not info.tab_hidden or via_link is not None
                result = (
                    parent_crumb + [section.label],
                    self._own_top.get(section.section_id) or parent_top,
                    parent_reachable and opens,
                )
            resolved[section.section_id] = result
            return result

        for s in self.sections:
            s.breadcrumb, s.top_trigger, s.reachable = resolve(s)

    def _register(self, section: Section) -> None:
        self.sections.append(section)
        self._buffers[id(section)] = []
        self._headings[id(section)] = []

    def _new_panel_section(self, tag: Tag, parent: Section) -> Section:
        info = self.panel_info.get(id(tag)) or _PanelInfo(tag["id"], tag["id"], None, False, False)
        self._id_counter[info.panel_id] += 1
        n = self._id_counter[info.panel_id]
        section_id = info.panel_id if n == 1 else f"{info.panel_id}~{n}"

        top_trigger = parent.top_trigger
        if info.trigger in self.top_triggers:
            self._own_top[section_id] = info.trigger
            top_trigger = top_trigger or info.trigger

        section = Section(
            section_id=section_id,
            dom_path=f"{parent.dom_path}/{section_id}",
            label=info.label,
            breadcrumb=parent.breadcrumb + [info.label],
            parent_id=parent.section_id,
            panel_chain=parent.panel_chain + [info.panel_id],
            hidden_by_default=parent.hidden_by_default or not info.active,
            ui_hidden=parent.ui_hidden or info.ui_hidden,
            top_trigger=top_trigger,
        )
        self._register(section)
        return section

    # ------------------------------------------------------------------ #

    def _heading_path(self, section: Section) -> list[str]:
        return [text for _, text in self._headings[id(section)]]

    def _set_heading(self, section: Section, level: int, text: str) -> None:
        stack = self._headings[id(section)]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))

    def _flush(self, section: Section) -> None:
        buffer = self._buffers[id(section)]
        if not buffer:
            return
        raw = "".join(buffer)
        buffer.clear()
        for pattern in self._noise:
            raw = pattern.sub(" ", raw)
        lines = []
        for line in normalize_text(raw).split("\n"):
            line = line.strip()
            if line in ("•", "-", "|", ">>", "»"):
                continue
            lines.append(line)
        text = normalize_text("\n".join(lines))
        if len(text.replace("•", "").strip()) >= 3:
            section.blocks.append(
                Block(kind="text", text=text, heading_path=self._heading_path(section))
            )

    def _walk(self, node: Tag, section: Section) -> None:
        self._walk_nodes(list(node.children), section)

    def _walk_nodes(self, nodes: list, section: Section) -> None:
        buffer = self._buffers[id(section)]
        for child in nodes:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in SKIP_TAGS:
                continue
            if _is_panel(child):
                self._flush(section)
                panel_section = self._new_panel_section(child, section)
                self._walk(child, panel_section)
                self._flush(panel_section)
                continue
            if name == "table":
                self._flush(section)
                rendered = render_table(
                    child,
                    glossary=self.profile.table_header_glossary,
                    drop_columns=self.profile.table_drop_columns,
                )
                if rendered.rows:
                    section.blocks.append(Block(
                        kind=rendered.kind,
                        text="\n".join(rendered.rows),
                        rows=rendered.rows if rendered.kind == "table" else [],
                        heading_path=self._heading_path(section),
                        overview=rendered.overview,
                    ))
                continue
            if name in HEADINGS:
                self._flush(section)
                # Malformed markup (e.g. <h1>…</h3>) can leave a heading open
                # around the whole article — then only its own text is the
                # heading and its element children are walked normally.
                wrapped = _has_block_children(child)
                text = _own_text(child) if wrapped else normalize_text(child.get_text(" ", strip=True))
                if text:
                    self._set_heading(section, int(name[1]), text)
                    buffer.append(text + "\n")
                if wrapped:
                    self._walk_nodes(child.find_all(True, recursive=False), section)
                continue
            if name == "br":
                buffer.append("\n")
            elif name == "li":
                buffer.append("\n• ")
                self._walk(child, section)
                buffer.append("\n")
            elif name in BLOCK_TAGS:
                buffer.append("\n")
                self._walk(child, section)
                buffer.append("\n")
            else:
                self._walk(child, section)


def _fallback_label(panel: Tag) -> str:
    """Label for a panel without a tab title: first heading / bold text /
    opening words of its own text (nested panels excluded)."""
    def own(tag: Tag) -> bool:
        return tag.find_parent(_is_panel) is panel

    for tag in panel.find_all(list(HEADINGS) + ["strong", "b"]):
        if own(tag):
            text = normalize_text(tag.get_text(" ", strip=True))
            if 3 <= len(text) <= 90:
                return text
    words = normalize_text(panel.get_text(" ", strip=True)).split()
    return " ".join(words[:8]) + ("…" if len(words) > 8 else "") if words else ""


def _panel_infos(root: Tag) -> dict[int, _PanelInfo]:
    """Label every panel from the tab list of ITS OWN tabs-content container."""
    infos: dict[int, _PanelInfo] = {}
    for panel in root.find_all(_is_panel):
        panel_id = panel["id"]
        ul = panel.parent.find_previous_sibling(_is_tabs_ul)
        link = ul.find("a", href=f"#{panel_id}") if ul is not None else None
        if link is None:
            link = root.find("a", href=f"#{panel_id}")
        label = normalize_text(link.get_text(" ", strip=True)) if link is not None else ""
        if not label:
            label = _fallback_label(panel) or panel_id
        li = link.find_parent("li") if link is not None else None
        infos[id(panel)] = _PanelInfo(
            panel_id=panel_id,
            label=label,
            trigger=link.get("id") if link is not None else None,
            active="active" in _classes(panel),
            ui_hidden=_style_hidden(li) or _style_hidden(link),
            tab_hidden=li is None or "hide" in _classes(li),
        )
    return infos


def _nav_parents(root: Tag, infos: dict[int, _PanelInfo]) -> dict[str, str]:
    """
    Panels whose tab title is hidden are opened by JS links elsewhere,
    e.g. "Stari programi: <a onclick="$('#tabela_osn_2016_click').click()">".
    Map such panel id → id of the panel holding such a link, so the
    breadcrumb follows what a visitor actually clicks (the DOM nesting
    sometimes puts old programs under unrelated tabs).  Hosts with a
    visible tab win over hidden / orphan ones; then document order.
    """
    trigger_to_panel: dict[str, str] = {}
    for a in root.find_all("a", href=True):
        if a["href"].startswith("#") and a.get("id"):
            trigger_to_panel.setdefault(a["id"], a["href"][1:])
    visible = {
        info.panel_id: not (info.tab_hidden or info.ui_hidden) for info in infos.values()
    }

    candidates: dict[str, list[tuple[bool, int, str]]] = {}
    for order, a in enumerate(root.find_all("a", onclick=True)):
        if a.find_parent("li", class_="tab-title") is not None:
            continue
        host = a.find_parent(_is_panel)
        if host is None:
            continue
        for trigger in _ONCLICK_TRIGGER_RE.findall(a["onclick"]):
            target = trigger_to_panel.get(trigger)
            if target and target != host["id"]:
                rank = (not visible.get(host["id"], False), order, host["id"])
                candidates.setdefault(target, []).append(rank)
    return {target: min(ranks)[2] for target, ranks in candidates.items()}


def _top_level_triggers(root: Tag) -> dict[str, str]:
    """Trigger ids of tab lists that are not inside any panel → panel id."""
    triggers: dict[str, str] = {}
    for ul in root.find_all(_is_tabs_ul):
        if ul.find_parent(_is_panel) is not None:
            continue
        for a in ul.find_all("a", href=True):
            href, trigger = a["href"], a.get("id")
            if href.startswith("#") and trigger:
                triggers[trigger] = href[1:]
    return triggers


def _parse_with_profile(
    soup: BeautifulSoup, url: str, final_url: str, profile: SiteProfile, allowed_domains
) -> Optional[Page]:
    root = None
    for selector in profile.content_selectors:
        candidate = soup.select_one(selector)
        if candidate is not None and len(candidate.get_text(strip=True)) > 50:
            root = candidate
            break
    if root is None:
        return None

    title = ""
    for selector in profile.title_selectors:
        tag = soup.select_one(selector)
        if tag is not None:
            title = _own_text(tag) if _has_block_children(tag) else normalize_text(tag.get_text(" ", strip=True))
            if title:
                break
    if not title and soup.title and soup.title.string:
        title = normalize_text(soup.title.string)
    title = title or url

    links, doc_links = _collect_links(root, final_url)
    panel_info = _panel_infos(root)
    top_triggers = _top_level_triggers(root)
    nav_parents = _nav_parents(root, panel_info)

    for selector in profile.drop_selectors:
        for tag in root.select(selector):
            tag.decompose()

    sections = _Walker(profile, title, panel_info, top_triggers, nav_parents).run(root)
    return Page(
        url=canonicalize_url(url, allowed_domains),
        display_url=display_url(final_url),
        doc_type="web",
        title=title,
        sections=sections,
        links=links,
        doc_links=doc_links,
        top_triggers=top_triggers,
    )


# --------------------------------------------------------------------------- #
#  Generic parser                                                              #
# --------------------------------------------------------------------------- #

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _parse_generic(html: str, soup: BeautifulSoup, url: str, final_url: str, allowed_domains) -> Page:
    import trafilatura

    markdown = None
    published = None
    title = ""
    try:
        markdown = trafilatura.extract(
            html, url=final_url, output_format="markdown", include_tables=True,
            include_links=False, include_images=False, include_comments=False,
            favor_recall=True,
        )
        meta = trafilatura.extract_metadata(html, default_url=final_url)
        if meta is not None:
            title = meta.title or ""
            published = meta.date
    except Exception as exc:  # trafilatura is best-effort
        logger.warning("trafilatura failed for %s: %s", url, exc)

    if not title and soup.title and soup.title.string:
        title = soup.title.string
    title = normalize_text(title) or url

    body = soup.body or soup
    links, doc_links = _collect_links(body, final_url)

    if not markdown:
        for tag in body.find_all(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        markdown = body.get_text("\n")

    sections: list[Section] = []
    current = Section(section_id="main", dom_path="main", label=title, breadcrumb=[title])
    sections.append(current)
    lines: list[str] = []

    def flush() -> None:
        text = normalize_text("\n".join(lines))
        if text:
            current.blocks.append(Block(kind="text", text=text))
        lines.clear()

    for line in markdown.split("\n"):
        match = _MD_HEADING_RE.match(line.strip())
        if match:
            flush()
            heading = normalize_text(match.group(2).strip("*_ "))
            if not heading:
                continue
            index = len(sections)
            current = Section(
                section_id=f"s{index}-{slugify(heading, 30)}",
                dom_path=f"md/{index}",
                label=heading,
                breadcrumb=[title, heading],
                parent_id="main",
            )
            sections.append(current)
            lines.append(heading)
        else:
            lines.append(line)
    flush()

    return Page(
        url=canonicalize_url(url, allowed_domains),
        display_url=display_url(final_url),
        doc_type="web",
        title=title,
        sections=sections,
        links=links,
        doc_links=doc_links,
        published_hint=published,
    )


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def parse_html(html: str, url: str, final_url: Optional[str] = None, allowed_domains=()) -> Page:
    final_url = final_url or url
    soup = BeautifulSoup(html, "lxml")
    profile = profile_for(final_url)
    if profile is not None:
        page = _parse_with_profile(soup, url, final_url, profile, allowed_domains)
        if page is not None and page.full_text.strip():
            return page
        soup = BeautifulSoup(html, "lxml")   # profile path mutates the tree
    return _parse_generic(html, soup, url, final_url, allowed_domains)
