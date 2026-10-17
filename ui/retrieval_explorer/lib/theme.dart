// theme.dart — colour tokens and text styles for the retrieval explorer.

import 'package:flutter/material.dart';

class Palette {
  static const bg = Color(0xFF0E1014);
  static const bgGradientTop = Color(0xFF131722);
  static const surface = Color(0xFF161A21);
  static const surfaceAlt = Color(0xFF1C2129);
  static const surfaceHigh = Color(0xFF232935);
  static const border = Color(0xFF262C38);
  static const borderStrong = Color(0xFF39414F);

  static const text = Color(0xFFE8EBF0);
  static const textSecondary = Color(0xFF9BA4B4);
  static const textMuted = Color(0xFF6B7280);

  static const accent = Color(0xFF7C9CFF);
  static const accentDim = Color(0xFF3B4A7A);
  static const success = Color(0xFF4ADE80);
  static const teal = Color(0xFF38BDF8);
  static const warn = Color(0xFFFBBF24);
  static const danger = Color(0xFFF87171);
  static const purple = Color(0xFFC084FC);

  /// Colour ramp for a cosine similarity score.
  static Color score(double similarity) {
    if (similarity >= 0.85) return success;
    if (similarity >= 0.78) return teal;
    if (similarity >= 0.70) return accent;
    if (similarity >= 0.60) return warn;
    return danger;
  }

  static Color reason(String? reason) {
    switch (reason) {
      case 'below_threshold':
        return warn;
      case 'legacy_or_outdated':
        return purple;
      case 'duplicate':
        return teal;
      case 'beyond_k':
        return textMuted;
      default:
        return textSecondary;
    }
  }
}

class Mono {
  static const family = 'Menlo';
  static const fallback = <String>['SF Mono', 'Monaco', 'Courier New', 'monospace'];
}

ThemeData buildTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: Palette.bg,
    canvasColor: Palette.bg,
    colorScheme: base.colorScheme.copyWith(
      primary: Palette.accent,
      secondary: Palette.teal,
      surface: Palette.surface,
      onSurface: Palette.text,
      error: Palette.danger,
      outline: Palette.border,
    ),
    dividerTheme: const DividerThemeData(color: Palette.border, thickness: 1, space: 1),
    textTheme: base.textTheme.apply(
      bodyColor: Palette.text,
      displayColor: Palette.text,
      fontFamily: '.AppleSystemUIFont',
    ),
    tooltipTheme: TooltipThemeData(
      decoration: BoxDecoration(
        color: Palette.surfaceHigh,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Palette.borderStrong),
      ),
      textStyle: const TextStyle(color: Palette.text, fontSize: 12),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: Palette.surfaceAlt,
      hintStyle: const TextStyle(color: Palette.textMuted, fontSize: 14),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: Palette.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: Palette.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: Palette.accent, width: 1.4),
      ),
    ),
    sliderTheme: const SliderThemeData(
      activeTrackColor: Palette.accent,
      inactiveTrackColor: Palette.surfaceHigh,
      thumbColor: Palette.accent,
      overlayColor: Color(0x337C9CFF),
      trackHeight: 3,
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith(
        (states) => states.contains(WidgetState.selected) ? Palette.accent : Palette.textMuted,
      ),
      trackColor: WidgetStateProperty.resolveWith(
        (states) => states.contains(WidgetState.selected)
            ? Palette.accentDim
            : Palette.surfaceHigh,
      ),
    ),
    scrollbarTheme: ScrollbarThemeData(
      thumbColor: WidgetStateProperty.all(Palette.borderStrong),
      thickness: WidgetStateProperty.all(8),
      radius: const Radius.circular(8),
    ),
  );
}
