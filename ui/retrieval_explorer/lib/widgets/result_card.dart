// result_card.dart — one retrieved chunk, fully structured.

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api_client.dart';
import '../theme.dart';
import 'common.dart';

class ResultCard extends StatefulWidget {
  const ResultCard({
    super.key,
    required this.hit,
    required this.terms,
    required this.threshold,
    required this.k,
  });

  final Hit hit;
  final List<String> terms;
  final double threshold;
  final int k;

  @override
  State<ResultCard> createState() => _ResultCardState();
}

class _ResultCardState extends State<ResultCard> {
  bool _fullText = false;
  bool _showMetadata = false;

  static const _previewChars = 380;

  @override
  Widget build(BuildContext context) {
    final hit = widget.hit;
    final color = Palette.score(hit.similarity);
    final text = _fullText ? hit.document : hit.body;
    final truncated = !_fullText && text.length > _previewChars;
    final shown = truncated ? '${text.substring(0, _previewChars)}…' : text;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: Palette.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: hit.rank == 1 ? color.withValues(alpha: 0.45) : Palette.border,
          width: hit.rank == 1 ? 1.3 : 1,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.22),
            blurRadius: 14,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ScoreBar(similarity: hit.similarity, threshold: widget.threshold, height: 4),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _header(hit),
                  const SizedBox(height: 10),
                  _tags(hit),
                  if (hit.url.isNotEmpty) ...[
                    const SizedBox(height: 10),
                    _urlRow(hit),
                  ],
                  const SizedBox(height: 12),
                  Container(height: 1, color: Palette.border),
                  const SizedBox(height: 12),
                  _body(hit, shown, truncated),
                  if (hit.chips.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: [
                        for (final chip in hit.chips)
                          MetaChip(label: chip.key, value: _short(chip.value)),
                      ],
                    ),
                  ],
                  const SizedBox(height: 6),
                  _metadataToggle(hit),
                  if (_showMetadata) _metadataTable(hit),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ---- sections -------------------------------------------------------- //

  Widget _header(Hit hit) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        RankBadge(rank: hit.rank, similarity: hit.similarity),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SelectableText(
                hit.title,
                maxLines: 2,
                style: const TextStyle(
                  color: Palette.text,
                  fontSize: 14.5,
                  fontWeight: FontWeight.w600,
                  height: 1.3,
                ),
              ),
              const SizedBox(height: 5),
              Row(
                children: [
                  Text(
                    'rank ${hit.rank} of ${widget.k}',
                    style: const TextStyle(color: Palette.textMuted, fontSize: 11),
                  ),
                  const _Dot(),
                  _mono('distance ${hit.distance.toStringAsFixed(4)}'),
                  const _Dot(),
                  _mono('${hit.words} words · ${hit.chars} chars'),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(width: 10),
        ScorePill(similarity: hit.similarity),
      ],
    );
  }

  Widget _tags(Hit hit) {
    final tags = <Widget>[
      Tag(label: hit.docKind, color: Palette.accent, icon: Icons.category_outlined),
      if (hit.status.isNotEmpty)
        Tag(
          label: hit.status,
          color: hit.isLegacy ? Palette.purple : Palette.success,
          icon: Icons.verified_outlined,
        ),
      if (hit.outdated) const Tag(label: 'outdated', color: Palette.warn, icon: Icons.history),
      if (hit.metadata['version_label'] != null)
        Tag(label: '${hit.metadata['version_label']}', color: Palette.teal),
      if (hit.metadata['block_type'] != null)
        Tag(label: '${hit.metadata['block_type']}', color: Palette.textSecondary),
      if (hit.metadata['embed_truncated'] == true)
        const Tag(label: 'truncated embedding', color: Palette.danger),
    ];
    return Wrap(spacing: 6, runSpacing: 6, children: tags);
  }

  Widget _urlRow(Hit hit) {
    return Container(
      padding: const EdgeInsets.fromLTRB(10, 2, 4, 2),
      decoration: BoxDecoration(
        color: Palette.surfaceAlt,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Palette.border),
      ),
      child: Row(
        children: [
          const Icon(Icons.link_rounded, size: 14, color: Palette.textMuted),
          const SizedBox(width: 8),
          Expanded(
            child: SelectableText(
              hit.url,
              maxLines: 1,
              style: const TextStyle(
                color: Palette.teal,
                fontSize: 11.5,
                fontFamily: Mono.family,
                fontFamilyFallback: Mono.fallback,
              ),
            ),
          ),
          CopyButton(text: hit.url, label: 'Copy link'),
          IconButton(
            tooltip: 'Open in browser',
            iconSize: 15,
            visualDensity: VisualDensity.compact,
            color: Palette.textSecondary,
            icon: const Icon(Icons.open_in_new_rounded),
            onPressed: () => _open(hit.url),
          ),
        ],
      ),
    );
  }

  Widget _body(Hit hit, String shown, bool truncated) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (_fullText)
          Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: Palette.accent.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(6),
            ),
            child: const Text(
              'embedded text — exactly what the vector was built from',
              style: TextStyle(color: Palette.accent, fontSize: 10.5),
            ),
          ),
        SelectableText.rich(
          TextSpan(
            children: highlightSpans(
              shown,
              widget.terms,
              base: const TextStyle(color: Palette.textSecondary, fontSize: 13, height: 1.55),
              hit: TextStyle(
                color: Palette.text,
                backgroundColor: Palette.accent.withValues(alpha: 0.22),
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
        Row(
          children: [
            if (hit.body.length > _previewChars || _fullText)
              TextButton.icon(
                onPressed: () => setState(() => _fullText = !_fullText),
                icon: Icon(
                  _fullText ? Icons.unfold_less_rounded : Icons.unfold_more_rounded,
                  size: 14,
                ),
                label: Text(
                  _fullText ? 'Collapse' : 'Show full chunk (${hit.chars} chars)',
                  style: const TextStyle(fontSize: 11.5),
                ),
                style: TextButton.styleFrom(
                  foregroundColor: Palette.accent,
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
              ),
            const Spacer(),
            CopyButton(text: hit.document, label: 'Copy chunk text'),
          ],
        ),
      ],
    );
  }

  Widget _metadataToggle(Hit hit) {
    return Align(
      alignment: Alignment.centerLeft,
      child: TextButton.icon(
        onPressed: () => setState(() => _showMetadata = !_showMetadata),
        icon: Icon(
          _showMetadata ? Icons.keyboard_arrow_down_rounded : Icons.keyboard_arrow_right_rounded,
          size: 16,
        ),
        label: Text(
          '${_showMetadata ? 'Hide' : 'Show'} metadata · chunk id ${_short(hit.id, 22)}',
          style: const TextStyle(fontSize: 11.5),
        ),
        style: TextButton.styleFrom(
          foregroundColor: Palette.textSecondary,
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          minimumSize: Size.zero,
          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        ),
      ),
    );
  }

  Widget _metadataTable(Hit hit) {
    final rows = hit.metadataRows;
    return Container(
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Palette.bg,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Palette.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                '${rows.length} metadata fields',
                style: const TextStyle(color: Palette.textMuted, fontSize: 11),
              ),
              const Spacer(),
              CopyButton(
                text: rows.map((e) => '${e.key}: ${e.value}').join('\n'),
                label: 'Copy metadata',
              ),
            ],
          ),
          const SizedBox(height: 6),
          for (final row in rows)
            KeyValueRow(
              label: row.key,
              value: row.value,
              mono: row.key == 'source' || row.key == 'content_hash' || row.key == 'dedup_key',
            ),
        ],
      ),
    );
  }

  // ---- helpers --------------------------------------------------------- //

  static Widget _mono(String value) => Text(
        value,
        style: const TextStyle(
          color: Palette.textMuted,
          fontSize: 11,
          fontFamily: Mono.family,
          fontFamilyFallback: Mono.fallback,
        ),
      );

  static String _short(String value, [int max = 46]) =>
      value.length <= max ? value : '${value.substring(0, max)}…';

  static Future<void> _open(String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null) return;
    try {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } catch (_) {
      // Sandboxed builds occasionally refuse; the link stays copyable.
    }
  }
}

class _Dot extends StatelessWidget {
  const _Dot();

  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(horizontal: 7),
        child: Text('·', style: TextStyle(color: Palette.textMuted, fontSize: 11)),
      );
}
