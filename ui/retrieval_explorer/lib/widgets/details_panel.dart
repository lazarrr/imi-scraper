// details_panel.dart — how the run was executed, and what the store contains.

import 'dart:convert';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';
import 'common.dart';

class RunDetailsPanel extends StatelessWidget {
  const RunDetailsPanel({super.key, required this.run});

  final SearchRun run;

  @override
  Widget build(BuildContext context) {
    final counts = run.counts;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      children: [
        const SectionTitle(title: 'Query', subtitle: 'What was embedded and searched'),
        const SizedBox(height: 10),
        Panel(
          child: Column(
            children: [
              KeyValueRow(label: 'as typed', value: run.query),
              KeyValueRow(label: 'normalised', value: run.normalizedQuery, mono: true),
              KeyValueRow(label: 'top-k', value: '${run.k}'),
              KeyValueRow(label: 'threshold', value: run.threshold.toStringAsFixed(2)),
              KeyValueRow(
                label: 'legacy included',
                value: run.includeLegacy ? 'yes' : 'no',
                valueColor: run.includeLegacy ? Palette.warn : null,
              ),
              KeyValueRow(
                label: 'metadata filter',
                value: run.whereFilter == null
                    ? 'none — all versions searched'
                    : const JsonEncoder.withIndent('  ').convert(run.whereFilter),
                mono: run.whereFilter != null,
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        const SectionTitle(
          title: 'Candidate funnel',
          subtitle: 'Every chunk Chroma returned, and what happened to it',
        ),
        const SizedBox(height: 10),
        Panel(
          child: Column(
            children: [
              _FunnelRow(
                label: 'returned by Chroma',
                value: counts.fetched,
                total: counts.fetched,
                color: Palette.accent,
                note: 'max(k, fetch_k) candidates',
              ),
              _FunnelRow(
                label: 'kept as results',
                value: counts.kept,
                total: counts.fetched,
                color: Palette.success,
              ),
              _FunnelRow(
                label: 'below threshold',
                value: counts.belowThreshold,
                total: counts.fetched,
                color: Palette.warn,
              ),
              _FunnelRow(
                label: 'legacy / outdated',
                value: counts.legacyOrOutdated,
                total: counts.fetched,
                color: Palette.purple,
              ),
              _FunnelRow(
                label: 'duplicates',
                value: counts.duplicate,
                total: counts.fetched,
                color: Palette.teal,
              ),
              _FunnelRow(
                label: 'beyond top-k',
                value: counts.beyondK,
                total: counts.fetched,
                color: Palette.textMuted,
              ),
              const SizedBox(height: 6),
              const Divider(),
              const SizedBox(height: 6),
              KeyValueRow(
                label: 'embedding',
                value: '${run.embedMs.toStringAsFixed(0)} ms  (E5 query encode)',
                mono: true,
              ),
              KeyValueRow(
                label: 'chroma query',
                value: '${run.queryMs.toStringAsFixed(0)} ms',
                mono: true,
              ),
              KeyValueRow(
                label: 'total',
                value: '${run.totalMs.toStringAsFixed(0)} ms',
                mono: true,
                valueColor: Palette.success,
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        _RawJson(raw: run.raw),
      ],
    );
  }
}

class _FunnelRow extends StatelessWidget {
  const _FunnelRow({
    required this.label,
    required this.value,
    required this.total,
    required this.color,
    this.note,
  });

  final String label;
  final int value;
  final int total;
  final Color color;
  final String? note;

  @override
  Widget build(BuildContext context) {
    final fraction = total == 0 ? 0.0 : value / total;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          SizedBox(
            width: 132,
            child: Text(label, style: const TextStyle(color: Palette.textSecondary, fontSize: 11.5)),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: SizedBox(
                height: 8,
                child: Stack(
                  children: [
                    Container(color: Palette.surfaceHigh),
                    FractionallySizedBox(
                      widthFactor: fraction.clamp(0.0, 1.0),
                      child: Container(color: color.withValues(alpha: 0.85)),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          SizedBox(
            width: 30,
            child: Text(
              '$value',
              textAlign: TextAlign.right,
              style: TextStyle(
                color: color,
                fontSize: 11.5,
                fontWeight: FontWeight.w700,
                fontFamily: Mono.family,
                fontFamilyFallback: Mono.fallback,
              ),
            ),
          ),
          if (note != null)
            Padding(
              padding: const EdgeInsets.only(left: 8),
              child: Text(note!, style: const TextStyle(color: Palette.textMuted, fontSize: 10)),
            ),
        ],
      ),
    );
  }
}

class StoreDetailsPanel extends StatelessWidget {
  const StoreDetailsPanel({
    super.key,
    required this.health,
    required this.stats,
    required this.onLoadStats,
    required this.statsLoading,
  });

  final HealthInfo? health;
  final StatsInfo? stats;
  final VoidCallback onLoadStats;
  final bool statsLoading;

  @override
  Widget build(BuildContext context) {
    if (health == null) {
      return const Center(
        child: Text('Not connected yet.', style: TextStyle(color: Palette.textMuted)),
      );
    }
    final store = health!.store;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      children: [
        const SectionTitle(title: 'Store', subtitle: 'The ChromaDB folder this project owns'),
        const SizedBox(height: 10),
        Panel(
          child: Column(
            children: [
              KeyValueRow(label: 'folder', value: store.path, mono: true),
              KeyValueRow(label: 'collection', value: store.collection),
              KeyValueRow(label: 'chunks', value: '${store.count}', mono: true),
              KeyValueRow(label: 'on disk', value: health!.sizeHuman, mono: true),
              KeyValueRow(label: 'embedding model', value: store.embeddingModel, mono: true),
              KeyValueRow(label: 'distance', value: store.distance, mono: true),
              KeyValueRow(label: 'query prefix', value: '"${store.queryPrefix}"', mono: true),
              KeyValueRow(label: 'passage prefix', value: '"${store.passagePrefix}"', mono: true),
              KeyValueRow(label: 'top-k / fetch-k',
                  value: '${store.threshold.toStringAsFixed(2)} threshold · fetch ${store.fetchK}'),
            ],
          ),
        ),
        if (store.collections.length > 1) ...[
          const SizedBox(height: 18),
          const SectionTitle(title: 'Collections in this folder'),
          const SizedBox(height: 10),
          Panel(
            child: Column(
              children: [
                for (final collection in store.collections)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      children: [
                        Icon(
                          collection.active
                              ? Icons.radio_button_checked_rounded
                              : Icons.radio_button_unchecked_rounded,
                          size: 13,
                          color: collection.active ? Palette.accent : Palette.textMuted,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            collection.name,
                            style: TextStyle(
                              color: collection.active ? Palette.text : Palette.textSecondary,
                              fontSize: 12,
                              fontFamily: Mono.family,
                              fontFamilyFallback: Mono.fallback,
                            ),
                          ),
                        ),
                        Text(
                          '${collection.count} chunks',
                          style: const TextStyle(color: Palette.textMuted, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],
        const SizedBox(height: 18),
        SectionTitle(
          title: 'Facets',
          subtitle: stats == null
              ? 'Counts across every chunk in the collection'
              : '${stats!.chunks} chunks · ${stats!.pages} pages',
          trailing: TextButton.icon(
            onPressed: statsLoading ? null : onLoadStats,
            icon: statsLoading
                ? const SizedBox(
                    width: 12,
                    height: 12,
                    child: CircularProgressIndicator(strokeWidth: 1.6),
                  )
                : const Icon(Icons.refresh_rounded, size: 14),
            label: Text(stats == null ? 'Load' : 'Reload',
                style: const TextStyle(fontSize: 11.5)),
            style: TextButton.styleFrom(foregroundColor: Palette.accent),
          ),
        ),
        const SizedBox(height: 10),
        if (stats == null)
          const Panel(
            child: Text(
              'Press Load to scan the collection metadata (reads only, no model needed).',
              style: TextStyle(color: Palette.textMuted, fontSize: 11.5),
            ),
          )
        else ...[
          for (final entry in stats!.facets.entries)
            if (entry.value.isNotEmpty) ...[
              Panel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      entry.key,
                      style: const TextStyle(
                        color: Palette.textSecondary,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    for (final facet in entry.value.entries)
                      _FacetBar(
                        label: facet.key,
                        value: facet.value,
                        max: entry.value.values.reduce((a, b) => a > b ? a : b),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 10),
            ],
          if (stats!.topPages.isNotEmpty)
            Panel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'pages by chunk count',
                    style: TextStyle(
                      color: Palette.textSecondary,
                      fontSize: 11.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  for (final page in stats!.topPages.take(12))
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Row(
                        children: [
                          SizedBox(
                            width: 34,
                            child: Text(
                              '${page.value}',
                              style: const TextStyle(
                                color: Palette.accent,
                                fontSize: 11,
                                fontFamily: Mono.family,
                                fontFamilyFallback: Mono.fallback,
                              ),
                            ),
                          ),
                          Expanded(
                            child: Text(
                              page.key.replaceFirst(RegExp(r'^https?://'), ''),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(color: Palette.textMuted, fontSize: 11),
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
        ],
      ],
    );
  }
}

class _FacetBar extends StatelessWidget {
  const _FacetBar({required this.label, required this.value, required this.max});

  final String label;
  final int value;
  final int max;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          SizedBox(
            width: 118,
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: Palette.textSecondary, fontSize: 11),
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: SizedBox(
                height: 7,
                child: Stack(
                  children: [
                    Container(color: Palette.surfaceHigh),
                    FractionallySizedBox(
                      widthFactor: max == 0 ? 0 : (value / max).clamp(0.0, 1.0),
                      child: Container(color: Palette.accent.withValues(alpha: 0.75)),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          SizedBox(
            width: 46,
            child: Text(
              '$value',
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: Palette.textSecondary,
                fontSize: 11,
                fontFamily: Mono.family,
                fontFamilyFallback: Mono.fallback,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _RawJson extends StatefulWidget {
  const _RawJson({required this.raw});

  final Map<String, dynamic> raw;

  @override
  State<_RawJson> createState() => _RawJsonState();
}

class _RawJsonState extends State<_RawJson> {
  bool _open = false;

  @override
  Widget build(BuildContext context) {
    final pretty = const JsonEncoder.withIndent('  ').convert(widget.raw);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: SectionTitle(
                title: 'Raw response',
                subtitle: '${pretty.length} chars of JSON — what the UI received',
              ),
            ),
            CopyButton(text: pretty, label: 'Copy JSON'),
          ],
        ),
        const SizedBox(height: 8),
        if (_open)
          Container(
            constraints: const BoxConstraints(maxHeight: 420),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Palette.bg,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Palette.border),
            ),
            child: SingleChildScrollView(
              child: SelectableText(
                pretty,
                style: const TextStyle(
                  color: Palette.textSecondary,
                  fontSize: 10.5,
                  height: 1.45,
                  fontFamily: Mono.family,
                  fontFamilyFallback: Mono.fallback,
                ),
              ),
            ),
          )
        else
          OutlinedButton(
            onPressed: () => setState(() => _open = true),
            style: OutlinedButton.styleFrom(
              foregroundColor: Palette.textSecondary,
              side: const BorderSide(color: Palette.border),
            ),
            child: const Text('Expand JSON', style: TextStyle(fontSize: 11.5)),
          ),
      ],
    );
  }
}
