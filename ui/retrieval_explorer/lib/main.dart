// main.dart — Chroma Retrieval Explorer
// ---------------------------------------------------------------------------
// Desktop UI for inspecting what the vector store actually returns: the top-k
// chunks with every score, badge and metadata field, plus the candidates the
// retrieval policy threw away.  It talks to `tools/retrieval_api.py`
// (`python run.py ui`), which owns the Chroma + E5 side of things.
// ---------------------------------------------------------------------------

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api_client.dart';
import 'theme.dart';
import 'widgets/common.dart';
import 'widgets/details_panel.dart';
import 'widgets/query_bar.dart';
import 'widgets/rejected_panel.dart';
import 'widgets/result_card.dart';

const List<String> kExampleQueries = [
  'Koliko traje master studije informatike?',
  'Uslovi za upis na osnovne akademske studije',
  'Školarina za samofinansirajuće studente',
  'Kontakt i adresa Fakulteta',
];

void main() {
  runApp(const RetrievalExplorerApp());
}

class RetrievalExplorerApp extends StatelessWidget {
  const RetrievalExplorerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Chroma Retrieval Explorer',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(),
      home: const ExplorerPage(),
    );
  }
}

enum _Backend { connecting, ready, unreachable }

class ExplorerPage extends StatefulWidget {
  const ExplorerPage({super.key, this.api});

  /// Injectable for tests / the screenshot harness; production uses the
  /// default client pointing at `http://127.0.0.1:8765`.
  final RetrievalApi? api;

  @override
  State<ExplorerPage> createState() => _ExplorerPageState();
}

class _ExplorerPageState extends State<ExplorerPage> {
  late final RetrievalApi _api = widget.api ?? RetrievalApi();
  late final bool _ownsApi = widget.api == null;
  final TextEditingController _queryController = TextEditingController();
  final FocusNode _queryFocus = FocusNode();

  _Backend _backend = _Backend.connecting;
  String? _backendMessage;
  HealthInfo? _health;

  SearchRun? _run;
  String? _searchError;
  bool _searching = false;

  StatsInfo? _stats;
  bool _statsLoading = false;

  int _k = 5;
  bool _includeLegacy = false;
  double? _thresholdOverride;
  String? _collection;
  int _railTab = 0;
  bool _autoRan = false;

  @override
  void initState() {
    super.initState();
    _queryController.addListener(() {
      if (mounted) setState(() {});
    });
    _connect();
  }

  @override
  void dispose() {
    if (_ownsApi) _api.dispose();
    _queryController.dispose();
    _queryFocus.dispose();
    super.dispose();
  }

  // ------------------------------------------------------------------ //
  //  Backend                                                            //
  // ------------------------------------------------------------------ //

  Future<void> _connect() async {
    setState(() {
      _backend = _Backend.connecting;
      _backendMessage = null;
    });
    final deadline = DateTime.now().add(const Duration(seconds: 30));
    while (mounted) {
      try {
        final health = await _api.health();
        if (!mounted) return;
        setState(() {
          _health = health;
          _backend = _Backend.ready;
          _backendMessage = health.modelError;
          final collections = health.store.collections;
          _collection = collections.isEmpty
              ? health.store.collection
              : collections
                  .firstWhere((c) => c.active, orElse: () => collections.first)
                  .name;
        });
        if (!_autoRan) {
          _autoRan = true;
          _runQuery(kExampleQueries.first);
        }
        return;
      } on ApiException catch (error) {
        if (!mounted) return;
        setState(() => _backendMessage = error.message);
        if (DateTime.now().isAfter(deadline)) break;
        await Future<void>.delayed(const Duration(milliseconds: 1200));
      }
    }
    if (mounted) setState(() => _backend = _Backend.unreachable);
  }

  Future<void> _runQuery(String query) async {
    final trimmed = query.trim();
    if (trimmed.isEmpty) return;
    _queryController.text = trimmed;
    _queryController.selection = TextSelection.collapsed(offset: trimmed.length);
    setState(() {
      _searching = true;
      _searchError = null;
    });
    try {
      final run = await _api.search(
        query: trimmed,
        k: _k,
        includeLegacy: _includeLegacy,
        threshold: _thresholdOverride,
        collection: _collection,
      );
      if (!mounted) return;
      setState(() => _run = run);
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _searchError = error.message);
    } finally {
      if (mounted) setState(() => _searching = false);
    }
  }

  Future<void> _loadStats() async {
    if (_statsLoading) return;
    setState(() => _statsLoading = true);
    try {
      final stats = await _api.stats(collection: _collection);
      if (!mounted) return;
      setState(() => _stats = stats);
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _searchError = error.message);
    } finally {
      if (mounted) setState(() => _statsLoading = false);
    }
  }

  // ------------------------------------------------------------------ //
  //  Build                                                              //
  // ------------------------------------------------------------------ //

  @override
  Widget build(BuildContext context) {
    final double effectiveThreshold =
        _thresholdOverride ?? _health?.store.threshold ?? 0.7;
    final terms = queryTerms(_queryController.text);

    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.keyR, meta: true): () =>
            _runQuery(_queryController.text),
        const SingleActivator(LogicalKeyboardKey.enter, meta: true): () =>
            _runQuery(_queryController.text),
      },
      child: Focus(
        autofocus: true,
        child: Scaffold(
          backgroundColor: Palette.bg,
          body: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Palette.bgGradientTop, Palette.bg],
                stops: [0.0, 0.35],
              ),
            ),
            child: SafeArea(
              child: Column(
                children: [
                  _Header(
                    health: _health,
                    backend: _backend,
                    message: _backendMessage,
                    onRetry: _connect,
                  ),
                  if (_backend == _Backend.unreachable)
                    Expanded(
                      child: _OfflineState(message: _backendMessage, onRetry: _connect),
                    )
                  else ...[
                    QueryBar(
                      controller: _queryController,
                      onSearch: () => _runQuery(_queryController.text),
                      loading: _searching,
                      k: _k,
                      onKChanged: (value) {
                        setState(() => _k = value);
                        if (_run != null) _runQuery(_queryController.text);
                      },
                      threshold: effectiveThreshold,
                      defaultThreshold: _health?.store.threshold ?? 0.7,
                      onThresholdChanged: (value) =>
                          setState(() => _thresholdOverride = value),
                      includeLegacy: _includeLegacy,
                      onIncludeLegacyChanged: (value) {
                        setState(() => _includeLegacy = value);
                        if (_run != null) _runQuery(_queryController.text);
                      },
                      collection: _collection,
                      collections: _health?.store.collections ?? const [],
                      onCollectionChanged: (value) {
                        setState(() {
                          _collection = value;
                          _stats = null;
                        });
                        if (_run != null) _runQuery(_queryController.text);
                      },
                      onClear: () {
                        _queryController.clear();
                        _queryFocus.requestFocus();
                      },
                    ),
                    Expanded(
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Expanded(child: _resultsPane(terms)),
                          _RightRail(
                            index: _railTab,
                            onChanged: (index) {
                              setState(() => _railTab = index);
                              if (index == 2 && _stats == null) _loadStats();
                            },
                            run: _run,
                            terms: terms,
                            health: _health,
                            stats: _stats,
                            statsLoading: _statsLoading,
                            onLoadStats: _loadStats,
                          ),
                        ],
                      ),
                    ),
                    _StatusBar(run: _run, searching: _searching),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _resultsPane(List<String> terms) {
    if (_searchError != null) {
      return _ErrorState(
        message: _searchError!,
        onRetry: () => _runQuery(_queryController.text),
      );
    }
    if (_searching && _run == null) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(strokeWidth: 2),
            SizedBox(height: 16),
            Text(
              'Embedding the query and reading the HNSW index…',
              style: TextStyle(color: Palette.textMuted, fontSize: 12.5),
            ),
          ],
        ),
      );
    }
    if (_run == null) return _EmptyState(onExample: _runQuery);

    final run = _run!;
    return Stack(
      children: [
        ListView.builder(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
          itemCount: run.results.length + 1,
          itemBuilder: (context, index) {
            if (index == 0) return _resultsHeader(run);
            final hit = run.results[index - 1];
            return ResultCard(
              key: ValueKey('${hit.id}-${run.query}'),
              hit: hit,
              terms: terms,
              threshold: run.threshold,
              k: run.k,
            );
          },
        ),
        if (_searching)
          const Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: LinearProgressIndicator(minHeight: 2, backgroundColor: Colors.transparent),
          ),
      ],
    );
  }

  Widget _resultsHeader(SearchRun run) {
    final counts = run.counts;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SectionTitle(
            title: 'Top ${run.results.length} of ${counts.fetched} candidates',
            subtitle: 'cosine similarity · threshold ${run.threshold.toStringAsFixed(2)} · '
                '${counts.rejected} filtered out · ${run.totalMs.toStringAsFixed(0)} ms',
            trailing: Row(
              children: [
                for (final entry in const [
                  ('≥85%', Palette.success),
                  ('≥78%', Palette.teal),
                  ('≥70%', Palette.accent),
                  ('≥60%', Palette.warn),
                  ('<60%', Palette.danger),
                ])
                  Padding(
                    padding: const EdgeInsets.only(left: 6),
                    child: Row(
                      children: [
                        Container(
                          width: 7,
                          height: 7,
                          decoration: BoxDecoration(
                            color: entry.$2,
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                        const SizedBox(width: 3),
                        Text(
                          entry.$1,
                          style: const TextStyle(color: Palette.textMuted, fontSize: 9.5),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Query: "${run.query}"  →  embedded as "${run.store.queryPrefix}${run.normalizedQuery}"',
            style: const TextStyle(color: Palette.textMuted, fontSize: 11),
          ),
          if (run.store.count == 0)
            Container(
              margin: const EdgeInsets.only(top: 8),
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              decoration: BoxDecoration(
                color: Palette.warn.withValues(alpha: 0.10),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Palette.warn.withValues(alpha: 0.35)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.warning_amber_rounded, size: 14, color: Palette.warn),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Collection "${run.store.collection}" is empty — run '
                      '`python run.py ingest`, or pick another collection above.',
                      style: const TextStyle(color: Palette.warn, fontSize: 11.5),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

// --------------------------------------------------------------------------- //
//  Chrome                                                                      //
// --------------------------------------------------------------------------- //

class _Header extends StatelessWidget {
  const _Header({
    required this.health,
    required this.backend,
    required this.message,
    required this.onRetry,
  });

  final HealthInfo? health;
  final _Backend backend;
  final String? message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final store = health?.store;
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 12),
      decoration: const BoxDecoration(
        color: Palette.bg,
        border: Border(bottom: BorderSide(color: Palette.border)),
      ),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [Palette.accent, Palette.purple]),
              borderRadius: BorderRadius.circular(9),
            ),
            child: const Icon(Icons.hub_rounded, size: 18, color: Color(0xFF0B0E14)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Chroma Retrieval Explorer',
                  style: TextStyle(
                    fontSize: 14.5,
                    fontWeight: FontWeight.w700,
                    color: Palette.text,
                    letterSpacing: 0.1,
                  ),
                ),
                Text(
                  store == null
                      ? 'waiting for the retrieval API'
                      : '${store.path}  ·  ${store.embeddingModel}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 10.5,
                    color: Palette.textMuted,
                    fontFamily: Mono.family,
                    fontFamilyFallback: Mono.fallback,
                  ),
                ),
              ],
            ),
          ),
          if (store != null) ...[
            _Stat(label: 'COLLECTION', value: store.collection),
            _Stat(label: 'CHUNKS', value: '${store.count}'),
            _Stat(label: 'ON DISK', value: health?.sizeHuman ?? '—'),
          ],
          const SizedBox(width: 8),
          _StatusChip(backend: backend, health: health, message: message, onRetry: onRetry),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(left: 8),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Palette.surfaceAlt,
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: Palette.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Palette.textMuted,
              fontSize: 9,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.5,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Palette.text,
              fontSize: 11.5,
              fontFamily: Mono.family,
              fontFamilyFallback: Mono.fallback,
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({
    required this.backend,
    required this.health,
    required this.message,
    required this.onRetry,
  });

  final _Backend backend;
  final HealthInfo? health;
  final String? message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    if (backend == _Backend.connecting) {
      return const Tag(label: 'connecting…', color: Palette.warn, icon: Icons.sync_rounded);
    }
    if (backend == _Backend.unreachable) {
      return InkWell(
        onTap: onRetry,
        child: const Tag(
          label: 'backend offline — retry',
          color: Palette.danger,
          icon: Icons.error_outline,
        ),
      );
    }
    if (health?.modelLoaded == false) {
      return Tooltip(
        message: health?.warming == true
            ? 'The backend is loading intfloat/multilingual-e5-large (2.1 GB). '
                'The first search waits for it.'
            : (message ?? 'Model not loaded yet'),
        child: const Tag(
          label: 'loading E5 model…',
          color: Palette.warn,
          icon: Icons.downloading_rounded,
        ),
      );
    }
    return const Tag(label: 'ready', color: Palette.success, icon: Icons.check_circle_outline);
  }
}

class _RightRail extends StatelessWidget {
  const _RightRail({
    required this.index,
    required this.onChanged,
    required this.run,
    required this.terms,
    required this.health,
    required this.stats,
    required this.statsLoading,
    required this.onLoadStats,
  });

  final int index;
  final ValueChanged<int> onChanged;
  final SearchRun? run;
  final List<String> terms;
  final HealthInfo? health;
  final StatsInfo? stats;
  final bool statsLoading;
  final VoidCallback onLoadStats;

  @override
  Widget build(BuildContext context) {
    final rejected = run?.rejected.length ?? 0;
    return Container(
      width: 400,
      decoration: const BoxDecoration(
        color: Color(0xFF11141A),
        border: Border(left: BorderSide(color: Palette.border)),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Palette.border)),
            ),
            child: Row(
              children: [
                _RailTab(
                  label: 'Filtered out',
                  badge: rejected,
                  selected: index == 0,
                  onTap: () => onChanged(0),
                ),
                _RailTab(label: 'Run', selected: index == 1, onTap: () => onChanged(1)),
                _RailTab(label: 'Store', selected: index == 2, onTap: () => onChanged(2)),
              ],
            ),
          ),
          Expanded(
            child: switch (index) {
              0 => run == null
                  ? const _RailPlaceholder()
                  : RejectedPanel(rejected: run!.rejected, terms: terms, counts: run!.counts),
              1 => run == null ? const _RailPlaceholder() : RunDetailsPanel(run: run!),
              _ => StoreDetailsPanel(
                  health: health,
                  stats: stats,
                  onLoadStats: onLoadStats,
                  statsLoading: statsLoading,
                ),
            },
          ),
        ],
      ),
    );
  }
}

class _RailPlaceholder extends StatelessWidget {
  const _RailPlaceholder();

  @override
  Widget build(BuildContext context) => const Center(
        child: Text(
          'Run a search first.',
          style: TextStyle(color: Palette.textMuted, fontSize: 12),
        ),
      );
}

class _RailTab extends StatelessWidget {
  const _RailTab({
    required this.label,
    required this.selected,
    required this.onTap,
    this.badge,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;
  final int? badge;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 3),
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: selected ? Palette.surfaceHigh : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: selected ? Palette.borderStrong : Colors.transparent),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                label,
                style: TextStyle(
                  color: selected ? Palette.text : Palette.textMuted,
                  fontSize: 11.5,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
              if (badge != null && badge! > 0) ...[
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                  decoration: BoxDecoration(
                    color: Palette.warn.withValues(alpha: 0.18),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '$badge',
                    style: const TextStyle(
                      color: Palette.warn,
                      fontSize: 9.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.run, required this.searching});

  final SearchRun? run;
  final bool searching;

  @override
  Widget build(BuildContext context) {
    final counts = run?.counts;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      decoration: const BoxDecoration(
        color: Palette.bg,
        border: Border(top: BorderSide(color: Palette.border)),
      ),
      child: Row(
        children: [
          Icon(
            searching ? Icons.hourglass_top_rounded : Icons.check_circle_outline,
            size: 13,
            color: searching ? Palette.warn : Palette.success,
          ),
          const SizedBox(width: 8),
          Text(
            run == null
                ? 'idle'
                : '${counts!.kept} results · ${counts.rejected} rejected · '
                    'embed ${run!.embedMs.toStringAsFixed(0)} ms · '
                    'chroma ${run!.queryMs.toStringAsFixed(0)} ms · '
                    'total ${run!.totalMs.toStringAsFixed(0)} ms',
            style: const TextStyle(
              color: Palette.textMuted,
              fontSize: 11,
              fontFamily: Mono.family,
              fontFamilyFallback: Mono.fallback,
            ),
          ),
          const Spacer(),
          const Text(
            'Enter to search  ·  ⌘R to re-run  ·  expand a card for full metadata',
            style: TextStyle(color: Palette.textMuted, fontSize: 10.5),
          ),
        ],
      ),
    );
  }
}

// --------------------------------------------------------------------------- //
//  Empty / error states                                                        //
// --------------------------------------------------------------------------- //

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.onExample});

  final ValueChanged<String> onExample;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.scatter_plot_rounded,
                size: 40, color: Palette.accent.withValues(alpha: 0.7)),
            const SizedBox(height: 16),
            const Text(
              'Ready to retrieve',
              style: TextStyle(color: Palette.text, fontSize: 16, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            const Text(
              'Type a question above, or start from one of these:',
              textAlign: TextAlign.center,
              style: TextStyle(color: Palette.textMuted, fontSize: 12.5, height: 1.5),
            ),
            const SizedBox(height: 16),
            Wrap(
              alignment: WrapAlignment.center,
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final example in kExampleQueries)
                  ActionChip(
                    label: Text(example, style: const TextStyle(fontSize: 11.5)),
                    backgroundColor: Palette.surfaceAlt,
                    side: const BorderSide(color: Palette.border),
                    onPressed: () => onExample(example),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _OfflineState extends StatelessWidget {
  const _OfflineState({required this.message, required this.onRetry});

  final String? message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 620),
        child: Panel(
          padding: const EdgeInsets.all(26),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Icon(Icons.power_off_rounded, color: Palette.danger, size: 20),
                  SizedBox(width: 10),
                  Text(
                    'Retrieval backend not reachable',
                    style: TextStyle(
                      color: Palette.text,
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              SelectableText(
                message ?? 'No answer from ${RetrievalApi().base}.',
                style: const TextStyle(color: Palette.danger, fontSize: 12, height: 1.5),
              ),
              const SizedBox(height: 16),
              const Text(
                'Start it from the project root, then press Retry:',
                style: TextStyle(color: Palette.textSecondary, fontSize: 12.5),
              ),
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Palette.bg,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Palette.border),
                ),
                child: const SelectableText(
                  'python run.py ui                 # backend + this app\n'
                  'python tools/retrieval_api.py    # backend only',
                  style: TextStyle(
                    color: Palette.success,
                    fontSize: 12,
                    height: 1.6,
                    fontFamily: Mono.family,
                    fontFamilyFallback: Mono.fallback,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: onRetry,
                    icon: const Icon(Icons.refresh_rounded, size: 16),
                    label: const Text('Retry connection'),
                    style: FilledButton.styleFrom(
                      backgroundColor: Palette.accent,
                      foregroundColor: const Color(0xFF0B0E14),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    'expecting ${RetrievalApi().base}',
                    style: const TextStyle(
                      color: Palette.textMuted,
                      fontSize: 11,
                      fontFamily: Mono.family,
                      fontFamilyFallback: Mono.fallback,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  const _ErrorState({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: Panel(
          padding: const EdgeInsets.all(22),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Icon(Icons.error_outline_rounded, color: Palette.danger, size: 18),
                  SizedBox(width: 8),
                  Text(
                    'Search failed',
                    style: TextStyle(
                      color: Palette.text,
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              SelectableText(
                message,
                style: const TextStyle(color: Palette.danger, fontSize: 12, height: 1.5),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh_rounded, size: 15),
                label: const Text('Try again'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Palette.accent,
                  side: const BorderSide(color: Palette.borderStrong),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
