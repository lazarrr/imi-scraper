// query_bar.dart — the search box and the retrieval knobs.

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';

class QueryBar extends StatelessWidget {
  const QueryBar({
    super.key,
    required this.controller,
    required this.onSearch,
    required this.loading,
    required this.k,
    required this.onKChanged,
    required this.threshold,
    required this.defaultThreshold,
    required this.onThresholdChanged,
    required this.includeLegacy,
    required this.onIncludeLegacyChanged,
    required this.collection,
    required this.collections,
    required this.onCollectionChanged,
    required this.onClear,
  });

  final TextEditingController controller;
  final VoidCallback onSearch;
  final bool loading;
  final int k;
  final ValueChanged<int> onKChanged;
  final double threshold;
  final double defaultThreshold;
  final ValueChanged<double> onThresholdChanged;
  final bool includeLegacy;
  final ValueChanged<bool> onIncludeLegacyChanged;
  final String? collection;
  final List<CollectionInfo> collections;
  final ValueChanged<String?> onCollectionChanged;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 14),
      decoration: const BoxDecoration(
        color: Palette.bg,
        border: Border(bottom: BorderSide(color: Palette.border)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: controller,
                  onSubmitted: (_) => onSearch(),
                  style: const TextStyle(fontSize: 14.5, color: Palette.text),
                  decoration: InputDecoration(
                    hintText: 'Ask a question, e.g. "Koliko traje master studije informatike?"',
                    prefixIcon: const Icon(Icons.search_rounded, size: 19, color: Palette.textMuted),
                    suffixIcon: controller.text.isEmpty
                        ? null
                        : IconButton(
                            tooltip: 'Clear',
                            icon: const Icon(Icons.close_rounded, size: 16),
                            color: Palette.textMuted,
                            onPressed: onClear,
                          ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              SizedBox(
                height: 48,
                child: FilledButton.icon(
                  onPressed: loading ? null : onSearch,
                  icon: loading
                      ? const SizedBox(
                          width: 15,
                          height: 15,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Palette.bg),
                        )
                      : const Icon(Icons.bolt_rounded, size: 18),
                  label: Text(
                    loading ? 'Searching' : 'Search',
                    style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13.5),
                  ),
                  style: FilledButton.styleFrom(
                    backgroundColor: Palette.accent,
                    foregroundColor: const Color(0xFF0B0E14),
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 18,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              _Labeled(
                label: 'results',
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<int>(
                    value: k,
                    isDense: true,
                    dropdownColor: Palette.surfaceHigh,
                    style: const TextStyle(color: Palette.text, fontSize: 12.5),
                    icon: const Icon(Icons.expand_more_rounded,
                        size: 16, color: Palette.textMuted),
                    items: const [3, 5, 8, 10, 20]
                        .map((value) => DropdownMenuItem(
                              value: value,
                              child: Text('$value',
                                  style: const TextStyle(
                                    fontFamily: Mono.family,
                                    fontFamilyFallback: Mono.fallback,
                                    fontSize: 12.5,
                                  )),
                            ))
                        .toList(),
                    onChanged: (value) => value == null ? null : onKChanged(value),
                  ),
                ),
              ),
              _Labeled(
                label: 'threshold ${threshold.toStringAsFixed(2)}',
                child: SizedBox(
                  width: 150,
                  child: Slider(
                    value: threshold.clamp(0.0, 1.0),
                    onChanged: onThresholdChanged,
                  ),
                ),
              ),
              Tooltip(
                message: 'Reset the similarity threshold to the backend default '
                    '(${defaultThreshold.toStringAsFixed(2)})',
                child: TextButton(
                  onPressed: () => onThresholdChanged(defaultThreshold),
                  style: TextButton.styleFrom(
                    foregroundColor: Palette.textMuted,
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                  child: const Text('reset', style: TextStyle(fontSize: 11)),
                ),
              ),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Switch(
                    value: includeLegacy,
                    onChanged: onIncludeLegacyChanged,
                  ),
                  const SizedBox(width: 6),
                  Tooltip(
                    message: 'Include chunks with status=legacy or outdated=true. '
                        'Questions mentioning an old year enable this automatically.',
                    child: Text(
                      'include legacy / outdated',
                      style: TextStyle(
                        color: includeLegacy ? Palette.warn : Palette.textSecondary,
                        fontSize: 12,
                        fontWeight: includeLegacy ? FontWeight.w600 : FontWeight.w400,
                      ),
                    ),
                  ),
                ],
              ),
              if (collections.length > 1)
                _Labeled(
                  label: 'collection',
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: collection ?? collections.firstWhere(
                        (c) => c.active,
                        orElse: () => collections.first,
                      ).name,
                      isDense: true,
                      dropdownColor: Palette.surfaceHigh,
                      style: const TextStyle(color: Palette.text, fontSize: 12.5),
                      icon: const Icon(Icons.expand_more_rounded,
                          size: 16, color: Palette.textMuted),
                      items: collections
                          .map((c) => DropdownMenuItem(
                                value: c.name,
                                child: Text(
                                  '${c.name}  (${c.count})',
                                  style: const TextStyle(
                                    fontSize: 12.5,
                                    fontFamily: Mono.family,
                                    fontFamilyFallback: Mono.fallback,
                                  ),
                                ),
                              ))
                          .toList(),
                      onChanged: onCollectionChanged,
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Labeled extends StatelessWidget {
  const _Labeled({required this.label, required this.child});

  final String label;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: Palette.textMuted,
            fontSize: 10.5,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(width: 8),
        child,
      ],
    );
  }
}
