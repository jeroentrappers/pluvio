import 'package:meta/meta.dart';

@immutable
final class RadarFrame {
  const RadarFrame({
    required this.timestamp,
    required this.tileUrlTemplate,
  });

  /// Wall-clock time the radar sweep represents.
  final DateTime timestamp;

  /// WMS URL template with `{z}/{x}/{y}` placeholders (we synthesise a tile
  /// layer by binding TIME=<timestamp> against the configured WMS endpoint).
  final String tileUrlTemplate;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is RadarFrame &&
          other.timestamp == timestamp &&
          other.tileUrlTemplate == tileUrlTemplate;

  @override
  int get hashCode => Object.hash(timestamp, tileUrlTemplate);
}

@immutable
final class RadarAnimation {
  const RadarAnimation({
    required this.frames,
    required this.referenceTime,
  });

  /// Frames sorted oldest → newest, including any forecast frames after [referenceTime].
  final List<RadarFrame> frames;

  /// "Now" — the boundary between observation and forecast.
  final DateTime referenceTime;

  bool get isEmpty => frames.isEmpty;

  RadarFrame get latest => frames.last;

  /// Index of the frame closest to [referenceTime], used as the initial position.
  int get currentIndex {
    var bestIndex = 0;
    var bestDelta = Duration(microseconds: 1 << 62);
    for (var i = 0; i < frames.length; i++) {
      final delta = frames[i].timestamp.difference(referenceTime).abs();
      if (delta < bestDelta) {
        bestDelta = delta;
        bestIndex = i;
      }
    }
    return bestIndex;
  }
}
