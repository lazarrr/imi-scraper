"""
site_profiles.py
~~~~~~~~~~~~~~~~
Site-specific knowledge, kept in one place.

The IMI site (imi.pmf.kg.ac.rs) is a custom PHP CMS with Foundation 5
tabs.  Everything below was verified on the live HTML:

  • main content lives in div.background_panel (news: div.article_basic)
  • study-program pages hold every program version in tab panels
    (div.tabs-content > div.content#panelX), many of them hidden
  • ?osnovne / ?master / ?doktorske (and *_u_akreditaciji) only make the
    server inject a JS click that opens a tab — same content otherwise.
    Which params work differs per page, so they are probe-verified
    (see Fetcher.probe_tab_params), never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SiteProfile:
    name: str
    domains: tuple[str, ...]
    content_selectors: tuple[str, ...]
    title_selectors: tuple[str, ...]
    drop_selectors: tuple[str, ...]
    view_param_re: str = r"$^"                       # params that only switch tabs
    tab_param_candidates: tuple[str, ...] = ()
    noise_patterns: tuple[str, ...] = ()            # regexes removed from text
    table_header_glossary: dict[str, str] = field(default_factory=dict)
    table_drop_columns: tuple[str, ...] = ()

    def is_view_param(self, key: str) -> bool:
        return bool(re.match(self.view_param_re, key))


IMI_PROFILE = SiteProfile(
    name="imi",
    domains=("imi.pmf.kg.ac.rs",),
    content_selectors=("div.background_panel", "div.article_basic"),
    title_selectors=("h1.custom-panel",),
    drop_selectors=(
        "script", "style", "noscript", "iframe", "form", "button", "svg",
        "#form-login", ".owl-carousel", ".icon-bar", "ul.tabs", ".pswp",
    ),
    view_param_re=r"^(osnovne|master|doktorske)(_[a-z0-9_]+)?$",
    tab_param_candidates=(
        "osnovne", "osnovne_u_akreditaciji",
        "master", "master_u_akreditaciji",
        "doktorske",
    ),
    noise_patterns=(r"pogledaj program\s*(>>|»|›)?",),
    table_header_glossary={
        "S.": "Semestar",
        "Š.": "Šifra",
        "p": "predavanja",
        "v": "vežbe",
        "don": "DON",
    },
    table_drop_columns=("R.br.",),
)

PROFILES: tuple[SiteProfile, ...] = (IMI_PROFILE,)


def profile_for(url: str) -> SiteProfile | None:
    host = (urlsplit(url).hostname or "").lower()
    for profile in PROFILES:
        if any(host == d or host.endswith("." + d) for d in profile.domains):
            return profile
    return None
