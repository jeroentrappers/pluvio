import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../domain/radar_animation.dart';
import 'precipitation_legend.dart';

/// Bar chart of per-frame precipitation rate with a relative time axis.
/// Each bar is one [RadarFrame]; the active frame (driven by the timeline
/// scrubber) is outlined. Tick labels along the bottom show the time offset
/// from now (`+1h`, `+3h`, ...).
///
/// The chart adapts to whatever frames the backend returns — uniform 10-min
/// spacing for nowcast, hourly for short, etc. — without code changes.
///
/// TODO(model): when the trained model emits per-lead uncertainty, render
/// it as a thin whisker behind each bar (low/high band). The hook is at
/// the spot marked "uncertainty whisker placeholder" below.
class ForecastChart extends StatelessWidget {
  const ForecastChart({
    super.key,
    required this.frames,
    required this.referenceTime,
    required this.currentIndex,
  });

  final List<RadarFrame> frames;
  final DateTime referenceTime;

  /// Index into [frames] for the bar that should be drawn highlighted.
  /// `null` (or out of range) draws no highlight — used when the scrubber
  /// is parked on a past observation frame that isn't on the chart.
  final int? currentIndex;

  @override
  Widget build(BuildContext context) {
    if (frames.isEmpty) return const SizedBox.shrink();
    final scheme = Theme.of(context).colorScheme;
    final maxRate = frames
        .map((f) => f.valueMmPerHour)
        .fold<double>(0, (m, v) => v > m ? v : m)
        .clamp(0.5, double.infinity);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      spacing: 4,
      children: [
        Expanded(
          child: LayoutBuilder(
            builder: (_, c) {
              final slot = c.maxWidth / frames.length;
              final barWidth = (slot * 0.7).clamp(2.0, 14.0);
              return Stack(
                children: [
                  for (var i = 0; i < frames.length; i++)
                    _Bar(
                      left: slot * i + (slot - barWidth) / 2,
                      width: barWidth,
                      height: (frames[i].valueMmPerHour / maxRate) * c.maxHeight,
                      color: PrecipitationPalette.of(frames[i].level, scheme),
                      highlighted: currentIndex != null && i == currentIndex,
                      outline: scheme.onSurface,
                    ),
                  // uncertainty whisker placeholder — overlay thin vertical
                  // lines per frame once `RadarFrame.uncertainty` lands.
                ],
              );
            },
          ),
        ),
        _TimeAxis(
          frames: frames,
          color: scheme.onSurfaceVariant,
        ),
      ],
    );
  }
}

class _Bar extends StatelessWidget {
  const _Bar({
    required this.left,
    required this.width,
    required this.height,
    required this.color,
    required this.highlighted,
    required this.outline,
  });

  final double left;
  final double width;
  final double height;
  final Color color;
  final bool highlighted;
  final Color outline;

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: left,
      bottom: 0,
      width: width,
      height: height < 1 ? 1 : height,
      child: Container(
        decoration: BoxDecoration(
          color: color,
          border: highlighted ? Border.all(color: outline, width: 1.5) : null,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(2)),
        ),
      ),
    );
  }
}

/// Bottom axis with at most ~5 evenly-spaced tick labels showing
/// device-local clock time (`14:50`). Absolute time is more useful for the
/// user than a relative offset ("will it rain at 3pm?" beats "in 47m").
class _TimeAxis extends StatelessWidget {
  const _TimeAxis({required this.frames, required this.color});

  final List<RadarFrame> frames;
  final Color color;

  static const _targetTicks = 5;

  @override
  Widget build(BuildContext context) {
    final style = Theme.of(context).textTheme.labelSmall?.copyWith(color: color);
    final fmt = DateFormat.Hm();
    return SizedBox(
      height: 14,
      child: LayoutBuilder(
        builder: (_, c) {
          final slot = c.maxWidth / frames.length;
          final step = (frames.length / _targetTicks).ceil().clamp(1, frames.length);
          return Stack(
            children: [
              for (var i = 0; i < frames.length; i += step)
                Positioned(
                  left: slot * i + slot / 2 - 18,
                  width: 36,
                  child: Center(
                    child: Text(
                      fmt.format(frames[i].timestamp.toLocal()),
                      style: style,
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}
