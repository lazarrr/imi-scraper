// rejected_panel.dart — candidates that did NOT make the top-k, and why.

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';
import 'common.dart';

const _ruleNotes = {
  'below_threshold':
      'Cosine similarity under the threshold — the chatbot would never see these.',
  'legacy_or_outdated':
      'Old programme versions (status=legacy) or chunks DeepSeek flagged outdated. '
          'Ask about a year before 2019 ("stari program", "akreditacija 2016") to include them.',
  'duplicate':
      'Same dedup_key as a higher-ranked chunk — identical text for the same version.',
  'beyond_k':
      'Passed every rule but ranked below the top-k cut-off. Raise "Results" to surface them.',
};

const _order = ['below_threshold', 'legacy_or_outdated', 'duplicate', 'beyond_k'];

class RejectedPanel extends StatelessWidget {
  const RejectedPanel({
    super.key,
    required this.rejected,
    required this.terms,
    required this.counts,
  });

  final List<Hit> rejected;
  final List<String> terms;
  final SearchCounts counts;

  @override
  Widget build(BuildContext context) {
    if (rejected.isEmpty) {
      return const _EmptyNote(
        icon: Icons.verified_rounded,
        text: 'Nothing was filtered out — every candidate reached the ranking.',
      );
    }

    final grouped = <String, List<Hit>>{};
    for (final hit in rejected) {
      grouped.putIfAbsent(hit.reason ?? 'other', () => []).add(hit);
    }
    final reasons = grouped.keys.toList()
      ..sort((a, b) => _order.indexOf(a).compareTo(_order.indexOf(b)));

    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      children: [
        SectionTitle(
          title: 'Filtered out',
          subtitle: '${counts.rejected} candidates from the ${counts.fetched}-chunk '
              'candidate pool did not reach the ${counts.kept} results',
        ),
        const SizedBox(height: 14),
        for (final reason in reasons) ...[
          _ReasonGroup(
            reason: reason,
            hits: grouped[reason]!,
            terms: terms,
          ),
          const SizedBox(height: 14),
        ],
      ],
    );
  }
}

class _ReasonGroup extends StatelessWidget {
  const _ReasonGroup({required this.reason, required this.hits, required this.terms});

  final String reason;
  final List<Hit> hits;
  final List<String> terms;

  @override
  Widget build(BuildContext context) {
    final color = Palette.reason(reason);
    final label = hits.first.reasonLabel.isEmpty ? reason : hits.first.reasonLabel;
    return Panel(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.filter_alt_off_rounded, size: 14, color: color),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  label,
                  style: TextStyle(color: color, fontSize: 12.5, fontWeight: FontWeight.w700),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${hits.length}',
                  style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            _ruleNotes[reason] ?? '',
            style: const TextStyle(color: Palette.textMuted, fontSize: 11, height: 1.45),
          ),
          const SizedBox(height: 10),
          for (final hit in hits) _RejectedRow(hit: hit, terms: terms),
        ],
      ),
    );
  }
}

class _RejectedRow extends StatelessWidget {
  const _RejectedRow({required this.hit, required this.terms});

  final Hit hit;
  final List<String> terms;

  @override
  Widget build(BuildContext context) {
    final snippet = hit.body.length > 160 ? '${hit.body.substring(0, 160)}…' : hit.body;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.fromLTRB(10, 8, 10, 9),
      decoration: BoxDecoration(
        color: Palette.surfaceAlt,
        borderRadius: BorderRadius.circular(9),
        border: Border.all(color: Palette.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              ScorePill(similarity: hit.similarity, compact: true),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  hit.title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Palette.textSecondary,
                    fontSize: 11.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          if (hit.reasonDetail.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 5),
              child: Text(
                hit.reasonDetail,
                style: TextStyle(
                  color: Palette.reason(hit.reason).withValues(alpha: 0.85),
                  fontSize: 10.5,
                  fontFamily: Mono.family,
                  fontFamilyFallback: Mono.fallback,
                ),
              ),
            ),
          if (snippet.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: SelectableText.rich(
                TextSpan(
                  children: highlightSpans(
                    snippet,
                    terms,
                    base: const TextStyle(color: Palette.textMuted, fontSize: 11.5, height: 1.45),
                    hit: const TextStyle(
                      color: Palette.textSecondary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _EmptyNote extends StatelessWidget {
  const _EmptyNote({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 28, color: Palette.success.withValues(alpha: 0.8)),
            const SizedBox(height: 12),
            Text(
              text,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Palette.textMuted, fontSize: 12.5, height: 1.5),
            ),
          ],
        ),
      ),
    );
  }
}
