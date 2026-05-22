import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../domain/radar_animation.dart';

/// Horizontal scrubber over the frames of a [RadarAnimation]. Designed to
/// stay legible at small sizes — the labels show only the active frame and
/// the boundary between observation/forecast.
class TimelineSlider extends StatelessWidget {
  const TimelineSlider({
    super.key,
    required this.animation,
    required this.currentIndex,
    required this.onChanged,
  });

  final RadarAnimation animation;
  final int currentIndex;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    if (animation.isEmpty) return const SizedBox.shrink();

    final frame = animation.frames[currentIndex];
    final isForecast = frame.timestamp.isAfter(animation.referenceTime);
    final scheme = Theme.of(context).colorScheme;
    final fmt = DateFormat.Hm();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            children: [
              Text(
                fmt.format(frame.timestamp.toLocal()),
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(width: 8),
              if (isForecast)
                _Pill(label: 'forecast', color: scheme.secondary),
              if (!isForecast)
                _Pill(label: 'observation', color: scheme.primary),
            ],
          ),
        ),
        Slider(
          value: currentIndex.toDouble(),
          min: 0,
          max: (animation.frames.length - 1).toDouble(),
          divisions: animation.frames.length - 1,
          onChanged: (v) => onChanged(v.round()),
        ),
      ],
    );
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: Theme.of(context)
            .textTheme
            .labelSmall
            ?.copyWith(color: color, fontWeight: FontWeight.w600),
      ),
    );
  }
}
