import 'package:latlong2/latlong.dart';
import 'package:meta/meta.dart';

/// Buckets the raw precipitation rate (mm/h) into a coarse band that the UI
/// uses for color coding and copy ("light", "moderate", ...). Thresholds
/// follow the WMO 1985 classification.
enum PrecipitationLevel {
  none,
  light,
  moderate,
  heavy,
  violent;

  static PrecipitationLevel fromMmPerHour(double mmPerHour) {
    if (mmPerHour <= 0) return PrecipitationLevel.none;
    if (mmPerHour < 2.5) return PrecipitationLevel.light;
    if (mmPerHour < 7.5) return PrecipitationLevel.moderate;
    if (mmPerHour < 50) return PrecipitationLevel.heavy;
    return PrecipitationLevel.violent;
  }
}

@immutable
final class RadarFrame {
  const RadarFrame({
    required this.timestamp,
    required this.imageUrl,
    required this.valueMmPerHour,
  });

  /// Wall-clock time the radar sweep represents (UTC).
  final DateTime timestamp;

  /// Fully-resolved pre-signed PNG URL for the composite frame.
  /// Display this as a transparent overlay on top of a base map.
  final String imageUrl;

  /// Precipitation rate at the queried location, in mm/h. The wire format
  /// gives mm/10min — `KmiRadarRepository` multiplies by 6.
  final double valueMmPerHour;

  PrecipitationLevel get level =>
      PrecipitationLevel.fromMmPerHour(valueMmPerHour);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is RadarFrame &&
          other.timestamp == timestamp &&
          other.imageUrl == imageUrl &&
          other.valueMmPerHour == valueMmPerHour;

  @override
  int get hashCode => Object.hash(timestamp, imageUrl, valueMmPerHour);
}

@immutable
final class RadarAnimation {
  const RadarAnimation({
    required this.frames,
    required this.referenceTime,
    required this.location,
  });

  /// Frames sorted oldest → newest. Each frame is timestamped; some are in
  /// the past (observation), the tail is the 2-hour forecast.
  final List<RadarFrame> frames;

  /// "Now" — the boundary between observation and forecast.
  final DateTime referenceTime;

  /// The lat/lon the per-frame `valueMmPerHour` is reported for.
  final LatLng location;

  bool get isEmpty => frames.isEmpty;

  RadarFrame get latest => frames.last;

  /// Index of the frame closest to [referenceTime] — used as the initial
  /// timeline position.
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

  /// True if any frame at or after [referenceTime] shows precipitation.
  bool get hasRainAhead => frames
      .where((f) => !f.timestamp.isBefore(referenceTime))
      .any((f) => f.level != PrecipitationLevel.none);

  /// First future frame with rain, or `null` if dry through the whole horizon.
  RadarFrame? get firstFutureRainFrame {
    for (final f in frames) {
      if (f.timestamp.isBefore(referenceTime)) continue;
      if (f.level != PrecipitationLevel.none) return f;
    }
    return null;
  }

  /// Minutes from `referenceTime` until the first future rain frame.
  /// 0 if it's raining now, `null` if dry across the whole horizon.
  int? get minutesUntilRain {
    final f = firstFutureRainFrame;
    if (f == null) return null;
    final delta = f.timestamp.difference(referenceTime).inMinutes;
    return delta < 0 ? 0 : delta;
  }
}
