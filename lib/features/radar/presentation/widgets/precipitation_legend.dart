import 'package:flutter/material.dart';

import '../../domain/radar_animation.dart';

/// Color ramp used for both the legend and the nowcast bar chart. Keeping it
/// in one place avoids drift between the two views of the same data.
abstract final class PrecipitationPalette {
  static Color of(PrecipitationLevel level, ColorScheme scheme) {
    return switch (level) {
      PrecipitationLevel.none => scheme.surfaceContainerHighest,
      PrecipitationLevel.light => const Color(0xFF9ECAE1),
      PrecipitationLevel.moderate => const Color(0xFF3182BD),
      PrecipitationLevel.heavy => const Color(0xFFFD8D3C),
      PrecipitationLevel.violent => const Color(0xFFE31A1C),
    };
  }
}

class PrecipitationLegend extends StatelessWidget {
  const PrecipitationLegend({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Wrap(
      spacing: 12,
      runSpacing: 6,
      children: [
        for (final level in PrecipitationLevel.values.where((l) => l != PrecipitationLevel.none))
          _LegendChip(
            color: PrecipitationPalette.of(level, scheme),
            label: _labelFor(level),
          ),
      ],
    );
  }

  String _labelFor(PrecipitationLevel level) {
    return switch (level) {
      PrecipitationLevel.none => '–',
      PrecipitationLevel.light => 'Light',
      PrecipitationLevel.moderate => 'Moderate',
      PrecipitationLevel.heavy => 'Heavy',
      PrecipitationLevel.violent => 'Violent',
    };
  }
}

class _LegendChip extends StatelessWidget {
  const _LegendChip({required this.color, required this.label});

  final Color color;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(3),
          ),
        ),
        const SizedBox(width: 6),
        Text(label, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }
}
