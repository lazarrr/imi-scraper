// common.dart — small shared widgets: score bar, badges, chips, highlighting.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme.dart';

// ---------------------------------------------------------------------------
//  Query-term highlighting
// ---------------------------------------------------------------------------

const _foldMap = {
  'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
  'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'D',
};

/// Diacritic-insensitive fold that keeps the string length identical, so
/// match offsets stay valid for the original text.
String foldAscii(String input) {
  final buffer = StringBuffer();
  for (final rune in input.runes) {
    final char = String.fromCharCode(rune);
    buffer.write(_foldMap[char] ?? char);
  }
  return buffer.toString();
}

/// Query words worth highlighting (length >= 3, unique, folded).
List<String> queryTerms(String query) {
  final seen = <String>{};
  final terms = <String>[];
  for (final raw in query.split(RegExp(r'[^\p{L}\p{N}]+', unicode: true))) {
    if (raw.length < 3) continue;
    final folded = foldAscii(raw).toLowerCase();
    if (folded.isEmpty || !seen.add(folded)) continue;
    terms.add(folded);
  }
  return terms;
}

/// Splits [text] into spans, emphasising every occurrence of a query term.
List<TextSpan> highlightSpans(
  String text,
  List<String> terms, {
  required TextStyle base,
  required TextStyle hit,
}) {
  if (terms.isEmpty || text.isEmpty) return [TextSpan(text: text, style: base)];
  final folded = foldAscii(text).toLowerCase();
  final spans = <TextSpan>[];
  var cursor = 0;
  while (cursor < text.length) {
    var bestAt = -1;
    var bestLen = 0;
    for (final term in terms) {
      final at = folded.indexOf(term, cursor);
      if (at < 0) continue;
      if (bestAt == -1 || at < bestAt || (at == bestAt && term.length > bestLen)) {
        bestAt = at;
        bestLen = term.length;
      }
    }
    if (bestAt < 0) {
      spans.add(TextSpan(text: text.substring(cursor), style: base));
      break;
    }
    if (bestAt > cursor) {
      spans.add(TextSpan(text: text.substring(cursor, bestAt), style: base));
    }
    spans.add(TextSpan(
      text: text.substring(bestAt, bestAt + bestLen),
      style: hit,
    ));
    cursor = bestAt + bestLen;
  }
  return spans;
}

// ---------------------------------------------------------------------------
//  Score presentation
// ---------------------------------------------------------------------------

class ScoreBar extends StatelessWidget {
  const ScoreBar({
    super.key,
    required this.similarity,
    this.threshold,
    this.height = 4,
  });

  final double similarity;
  final double? threshold;
  final double height;

  @override
  Widget build(BuildContext context) {
    final color = Palette.score(similarity);
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        return SizedBox(
          height: height,
          child: Stack(
            children: [
              Container(color: Palette.surfaceHigh),
              AnimatedContainer(
                duration: const Duration(milliseconds: 260),
                curve: Curves.easeOutCubic,
                width: width * similarity.clamp(0.0, 1.0),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [color.withValues(alpha: 0.55), color],
                  ),
                ),
              ),
              if (threshold != null)
                Positioned(
                  left: (width * threshold!.clamp(0.0, 1.0)) - 1,
                  top: 0,
                  bottom: 0,
                  child: Container(width: 2, color: Palette.text.withValues(alpha: 0.55)),
                ),
            ],
          ),
        );
      },
    );
  }
}

class ScorePill extends StatelessWidget {
  const ScorePill({super.key, required this.similarity, this.compact = false});

  final double similarity;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final color = Palette.score(similarity);
    return Container(
      padding: EdgeInsets.symmetric(horizontal: compact ? 7 : 10, vertical: compact ? 3 : 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: color.withValues(alpha: 0.45)),
      ),
      child: Text(
        '${(similarity * 100).toStringAsFixed(1)}%',
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.w700,
          fontSize: compact ? 11.5 : 14,
          fontFamily: Mono.family,
          fontFamilyFallback: Mono.fallback,
          letterSpacing: -0.2,
        ),
      ),
    );
  }
}

class RankBadge extends StatelessWidget {
  const RankBadge({super.key, required this.rank, required this.similarity});

  final int rank;
  final double similarity;

  @override
  Widget build(BuildContext context) {
    final color = Palette.score(similarity);
    return Container(
      width: 40,
      height: 40,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.13),
        border: Border.all(color: color.withValues(alpha: 0.5)),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        '#$rank',
        style: TextStyle(color: color, fontWeight: FontWeight.w800, fontSize: 14),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
//  Labels
// ---------------------------------------------------------------------------

class Tag extends StatelessWidget {
  const Tag({super.key, required this.label, this.color, this.icon});

  final String label;
  final Color? color;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    final tone = color ?? Palette.textSecondary;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: tone.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: tone.withValues(alpha: 0.35)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 11, color: tone),
            const SizedBox(width: 4),
          ],
          Text(
            label,
            style: TextStyle(color: tone, fontSize: 11, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

class MetaChip extends StatelessWidget {
  const MetaChip({super.key, required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
      decoration: BoxDecoration(
        color: Palette.surfaceAlt,
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: Palette.border),
      ),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: '$label ',
              style: const TextStyle(
                color: Palette.textMuted,
                fontSize: 10.5,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.2,
              ),
            ),
            TextSpan(
              text: value,
              style: const TextStyle(color: Palette.textSecondary, fontSize: 11.5),
            ),
          ],
        ),
      ),
    );
  }
}

class SectionTitle extends StatelessWidget {
  const SectionTitle({super.key, required this.title, this.trailing, this.subtitle});

  final String title;
  final String? subtitle;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title.toUpperCase(),
                style: const TextStyle(
                  color: Palette.textSecondary,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.9,
                ),
              ),
              if (subtitle != null)
                Padding(
                  padding: const EdgeInsets.only(top: 2),
                  child: Text(
                    subtitle!,
                    style: const TextStyle(color: Palette.textMuted, fontSize: 11.5),
                  ),
                ),
            ],
          ),
        ),
        ?trailing,
      ],
    );
  }
}

/// Small copy-to-clipboard button that confirms with a toast-like tooltip.
class CopyButton extends StatefulWidget {
  const CopyButton({super.key, required this.text, this.label = 'Copy', this.icon});

  final String text;
  final String label;
  final IconData? icon;

  @override
  State<CopyButton> createState() => _CopyButtonState();
}

class _CopyButtonState extends State<CopyButton> {
  bool _copied = false;

  Future<void> _copy() async {
    await Clipboard.setData(ClipboardData(text: widget.text));
    if (!mounted) return;
    setState(() => _copied = true);
    await Future<void>.delayed(const Duration(milliseconds: 1400));
    if (mounted) setState(() => _copied = false);
  }

  @override
  Widget build(BuildContext context) {
    return TextButton.icon(
      onPressed: _copy,
      icon: Icon(
        _copied ? Icons.check_rounded : (widget.icon ?? Icons.copy_rounded),
        size: 14,
        color: _copied ? Palette.success : Palette.textSecondary,
      ),
      label: Text(
        _copied ? 'Copied' : widget.label,
        style: TextStyle(
          color: _copied ? Palette.success : Palette.textSecondary,
          fontSize: 11.5,
        ),
      ),
      style: TextButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        minimumSize: Size.zero,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
      ),
    );
  }
}

class Panel extends StatelessWidget {
  const Panel({super.key, required this.child, this.padding = const EdgeInsets.all(16)});

  final Widget child;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: Palette.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Palette.border),
      ),
      child: child,
    );
  }
}

class KeyValueRow extends StatelessWidget {
  const KeyValueRow({
    super.key,
    required this.label,
    required this.value,
    this.mono = false,
    this.valueColor,
  });

  final String label;
  final String value;
  final bool mono;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 132,
            child: Text(
              label,
              style: const TextStyle(color: Palette.textMuted, fontSize: 11.5),
            ),
          ),
          Expanded(
            child: SelectableText(
              value.isEmpty ? '—' : value,
              style: TextStyle(
                color: valueColor ?? Palette.text,
                fontSize: 11.5,
                fontFamily: mono ? Mono.family : null,
                fontFamilyFallback: mono ? Mono.fallback : null,
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
