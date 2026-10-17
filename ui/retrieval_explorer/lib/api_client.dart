// api_client.dart
// ---------------------------------------------------------------------------
// Thin client for the local retrieval API (`python tools/retrieval_api.py` /
// `python run.py ui`). Every model below mirrors a key of the JSON that
// `src/core/retrieval_service.py` returns, so the UI shows exactly what the
// CLI and the chatbot see.
// ---------------------------------------------------------------------------

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

/// Default backend address; override at build time with
/// `flutter run --dart-define=IMI_API=http://127.0.0.1:9000`.
const String kDefaultApiBase = String.fromEnvironment(
  'IMI_API',
  defaultValue: 'http://127.0.0.1:8765',
);

class ApiException implements Exception {
  ApiException(this.message, {this.status});

  final String message;
  final int? status;

  @override
  String toString() => message;
}

// ---------------------------------------------------------------------------
//  Models
// ---------------------------------------------------------------------------

class CollectionInfo {
  CollectionInfo({required this.name, required this.count, required this.active});

  final String name;
  final int count;
  final bool active;

  factory CollectionInfo.fromJson(Map<String, dynamic> json) => CollectionInfo(
        name: json['name'] as String? ?? '?',
        count: (json['count'] as num?)?.toInt() ?? -1,
        active: json['active'] as bool? ?? false,
      );
}

class StoreInfo {
  StoreInfo({
    required this.path,
    required this.collection,
    required this.count,
    required this.embeddingModel,
    required this.distance,
    required this.queryPrefix,
    required this.passagePrefix,
    required this.threshold,
    required this.fetchK,
    required this.collections,
  });

  final String path;
  final String collection;
  final int count;
  final String embeddingModel;
  final String distance;
  final String queryPrefix;
  final String passagePrefix;
  final double threshold;
  final int fetchK;
  final List<CollectionInfo> collections;

  factory StoreInfo.fromJson(Map<String, dynamic> json) => StoreInfo(
        path: json['path'] as String? ?? '',
        collection: json['collection'] as String? ?? '',
        count: (json['count'] as num?)?.toInt() ?? 0,
        embeddingModel: json['embedding_model'] as String? ?? '',
        distance: json['distance'] as String? ?? 'cosine',
        queryPrefix: json['query_prefix'] as String? ?? 'query: ',
        passagePrefix: json['passage_prefix'] as String? ?? 'passage: ',
        threshold: (json['threshold'] as num?)?.toDouble() ?? 0.7,
        fetchK: (json['fetch_k'] as num?)?.toInt() ?? 20,
        collections: ((json['collections'] as List?) ?? const [])
            .map((e) => CollectionInfo.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}

class HealthInfo {
  HealthInfo({
    required this.modelLoaded,
    required this.modelError,
    required this.warming,
    required this.uptimeSeconds,
    required this.store,
    required this.sizeBytes,
    required this.sizeHuman,
    required this.projectRoot,
  });

  final bool modelLoaded;
  final String? modelError;
  final bool warming;
  final double uptimeSeconds;
  final StoreInfo store;
  final int sizeBytes;
  final String sizeHuman;
  final String projectRoot;

  factory HealthInfo.fromJson(Map<String, dynamic> json) => HealthInfo(
        modelLoaded: json['model_loaded'] as bool? ?? false,
        modelError: json['model_error'] as String?,
        warming: json['warming'] as bool? ?? false,
        uptimeSeconds: (json['uptime_s'] as num?)?.toDouble() ?? 0,
        store: StoreInfo.fromJson((json['store'] as Map<String, dynamic>?) ?? const {}),
        sizeBytes: (json['size_bytes'] as num?)?.toInt() ?? 0,
        sizeHuman: json['size_human'] as String? ?? '—',
        projectRoot: json['project_root'] as String? ?? '',
      );
}

class Chip {
  Chip({required this.key, required this.value});

  final String key;
  final String value;

  factory Chip.fromJson(Map<String, dynamic> json) =>
      Chip(key: json['key'] as String? ?? '', value: json['value'] as String? ?? '');
}

class Hit {
  Hit({
    required this.id,
    required this.rank,
    required this.document,
    required this.body,
    required this.similarity,
    required this.distance,
    required this.metadata,
    required this.url,
    required this.breadcrumb,
    required this.pageTitle,
    required this.docKind,
    required this.status,
    required this.chips,
    required this.chars,
    required this.words,
    this.reason,
    this.reasonLabel = '',
    this.reasonDetail = '',
  });

  final String id;
  final int rank;
  final String document;
  final String body;
  final double similarity;
  final double distance;
  final Map<String, dynamic> metadata;
  final String url;
  final String breadcrumb;
  final String pageTitle;
  final String docKind;
  final String status;
  final List<Chip> chips;
  final int chars;
  final int words;
  final String? reason;
  final String reasonLabel;
  final String reasonDetail;

  bool get outdated => metadata['outdated'] == true;
  bool get isLegacy => status == 'legacy';

  String get title {
    if (breadcrumb.isNotEmpty) return breadcrumb;
    if (pageTitle.isNotEmpty) return pageTitle;
    return url.isNotEmpty ? url : id;
  }

  /// Ordered key/value pairs for the metadata drill-down.
  List<MapEntry<String, String>> get metadataRows {
    final keys = metadata.keys.toList()..sort();
    return keys
        .map((k) => MapEntry(k, _stringify(metadata[k])))
        .where((e) => e.value.isNotEmpty)
        .toList();
  }

  static String _stringify(Object? value) {
    if (value == null) return '';
    if (value is bool) return value ? 'true' : 'false';
    if (value is double) return value.toStringAsFixed(4);
    return value.toString();
  }

  factory Hit.fromJson(Map<String, dynamic> json) => Hit(
        id: json['id'] as String? ?? '',
        rank: (json['rank'] as num?)?.toInt() ?? 0,
        document: json['document'] as String? ?? '',
        body: json['body'] as String? ?? '',
        similarity: (json['similarity'] as num?)?.toDouble() ?? 0,
        distance: (json['distance'] as num?)?.toDouble() ?? 0,
        metadata: (json['metadata'] as Map<String, dynamic>?) ?? const {},
        url: json['url'] as String? ?? '',
        breadcrumb: json['breadcrumb'] as String? ?? '',
        pageTitle: json['page_title'] as String? ?? '',
        docKind: json['doc_kind'] as String? ?? 'chunk',
        status: json['status'] as String? ?? '',
        chips: ((json['chips'] as List?) ?? const [])
            .map((e) => Chip.fromJson(e as Map<String, dynamic>))
            .toList(),
        chars: (json['chars'] as num?)?.toInt() ?? 0,
        words: (json['words'] as num?)?.toInt() ?? 0,
        reason: json['reason'] as String?,
        reasonLabel: json['reason_label'] as String? ?? '',
        reasonDetail: json['reason_detail'] as String? ?? '',
      );
}

class SearchCounts {
  SearchCounts({
    required this.fetched,
    required this.kept,
    required this.rejected,
    required this.belowThreshold,
    required this.legacyOrOutdated,
    required this.duplicate,
    required this.beyondK,
  });

  final int fetched;
  final int kept;
  final int rejected;
  final int belowThreshold;
  final int legacyOrOutdated;
  final int duplicate;
  final int beyondK;

  factory SearchCounts.fromJson(Map<String, dynamic> json) => SearchCounts(
        fetched: (json['fetched'] as num?)?.toInt() ?? 0,
        kept: (json['kept'] as num?)?.toInt() ?? 0,
        rejected: (json['rejected'] as num?)?.toInt() ?? 0,
        belowThreshold: (json['below_threshold'] as num?)?.toInt() ?? 0,
        legacyOrOutdated: (json['legacy_or_outdated'] as num?)?.toInt() ?? 0,
        duplicate: (json['duplicate'] as num?)?.toInt() ?? 0,
        beyondK: (json['beyond_k'] as num?)?.toInt() ?? 0,
      );
}

class SearchRun {
  SearchRun({
    required this.query,
    required this.normalizedQuery,
    required this.k,
    required this.threshold,
    required this.includeLegacy,
    required this.whereFilter,
    required this.store,
    required this.embedMs,
    required this.queryMs,
    required this.totalMs,
    required this.counts,
    required this.results,
    required this.rejected,
    required this.raw,
  });

  final String query;
  final String normalizedQuery;
  final int k;
  final double threshold;
  final bool includeLegacy;
  final Object? whereFilter;
  final StoreInfo store;
  final double embedMs;
  final double queryMs;
  final double totalMs;
  final SearchCounts counts;
  final List<Hit> results;
  final List<Hit> rejected;
  final Map<String, dynamic> raw;

  factory SearchRun.fromJson(Map<String, dynamic> json) {
    final timing = (json['timing'] as Map<String, dynamic>?) ?? const {};
    return SearchRun(
      query: json['query'] as String? ?? '',
      normalizedQuery: json['normalized_query'] as String? ?? '',
      k: (json['k'] as num?)?.toInt() ?? 5,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.7,
      includeLegacy: json['include_legacy'] as bool? ?? false,
      whereFilter: json['where_filter'],
      store: StoreInfo.fromJson((json['store'] as Map<String, dynamic>?) ?? const {}),
      embedMs: (timing['embed_ms'] as num?)?.toDouble() ?? 0,
      queryMs: (timing['query_ms'] as num?)?.toDouble() ?? 0,
      totalMs: (timing['total_ms'] as num?)?.toDouble() ?? 0,
      counts: SearchCounts.fromJson((json['counts'] as Map<String, dynamic>?) ?? const {}),
      results: ((json['results'] as List?) ?? const [])
          .map((e) => Hit.fromJson(e as Map<String, dynamic>))
          .toList(),
      rejected: ((json['rejected'] as List?) ?? const [])
          .map((e) => Hit.fromJson(e as Map<String, dynamic>))
          .toList(),
      raw: json,
    );
  }
}

class StatsInfo {
  StatsInfo({required this.chunks, required this.pages, required this.facets, required this.topPages});

  final int chunks;
  final int pages;
  final Map<String, Map<String, int>> facets;
  final List<MapEntry<String, int>> topPages;

  factory StatsInfo.fromJson(Map<String, dynamic> json) {
    final facets = <String, Map<String, int>>{};
    ((json['facets'] as Map<String, dynamic>?) ?? const {}).forEach((key, value) {
      final inner = <String, int>{};
      ((value as Map<String, dynamic>?) ?? const {}).forEach((k, v) {
        inner[k] = (v as num).toInt();
      });
      facets[key] = inner;
    });
    return StatsInfo(
      chunks: (json['chunks'] as num?)?.toInt() ?? 0,
      pages: (json['pages'] as num?)?.toInt() ?? 0,
      facets: facets,
      topPages: (((json['top_pages'] as List?) ?? const [])
              .map((e) => MapEntry(
                    (e as Map<String, dynamic>)['url'] as String? ?? '—',
                    (e['chunks'] as num?)?.toInt() ?? 0,
                  ))
              .toList()),
    );
  }
}

// ---------------------------------------------------------------------------
//  Client
// ---------------------------------------------------------------------------

class RetrievalApi {
  RetrievalApi({String? baseUrl, http.Client? client})
      : base = Uri.parse(baseUrl ?? kDefaultApiBase),
        _client = client ?? http.Client();

  final Uri base;
  final http.Client _client;

  Uri _uri(String path) => base.replace(path: path);

  Future<Map<String, dynamic>> _decode(http.Response response) async {
    Map<String, dynamic> json;
    try {
      json = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    } on FormatException {
      throw ApiException('Backend sent a non-JSON reply (HTTP ${response.statusCode}).');
    }
    if (response.statusCode >= 400 || json['ok'] == false) {
      throw ApiException(
        json['error'] as String? ?? 'HTTP ${response.statusCode}',
        status: response.statusCode,
      );
    }
    return json;
  }

  Future<HealthInfo> health({Duration timeout = const Duration(seconds: 8)}) async {
    try {
      final response = await _client.get(_uri('/api/health')).timeout(timeout);
      return HealthInfo.fromJson(await _decode(response));
    } on SocketException catch (e) {
      throw ApiException('Cannot reach ${base.host}:${base.port} — ${e.osError?.message ?? 'connection refused'}');
    } on TimeoutException {
      throw ApiException('The backend did not answer within ${timeout.inSeconds}s.');
    }
  }

  Future<SearchRun> search({
    required String query,
    required int k,
    required bool includeLegacy,
    double? threshold,
    String? collection,
    Duration timeout = const Duration(minutes: 10),
  }) async {
    final response = await _client
        .post(
          _uri('/api/search'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({
            'query': query,
            'k': k,
            'include_legacy': includeLegacy,
            'threshold': ?threshold,
            if (collection != null && collection.isNotEmpty) 'collection': collection,
          }),
        )
        .timeout(timeout);
    return SearchRun.fromJson(await _decode(response));
  }

  Future<StatsInfo> stats({String? collection}) async {
    final uri = collection == null || collection.isEmpty
        ? _uri('/api/stats')
        : _uri('/api/stats').replace(queryParameters: {'collection': collection});
    return StatsInfo.fromJson(await _decode(await _client.get(uri)));
  }

  Future<Map<String, dynamic>> chunk(String id) async {
    return _decode(await _client.get(_uri('/api/chunk/$id')));
  }

  Future<void> warm() async {
    await _client.post(_uri('/api/warm'), body: '{}');
  }

  void dispose() => _client.close();
}
