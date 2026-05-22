import 'package:latlong2/latlong.dart';
import 'package:meta/meta.dart';

/// Buckets the raw precipitation intensity (mm/h) into a coarse band that the
/// UI uses for color coding and copy ("light", "moderate", ...). Thresholds
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
final class NowcastPoint {
  const NowcastPoint({
    required this.timestamp,
    required this.precipitationMmPerHour,
  });

  final DateTime timestamp;
  final double precipitationMmPerHour;

  PrecipitationLevel get level =>
      PrecipitationLevel.fromMmPerHour(precipitationMmPerHour);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is NowcastPoint &&
          other.timestamp == timestamp &&
          other.precipitationMmPerHour == precipitationMmPerHour;

  @override
  int get hashCode => Object.hash(timestamp, precipitationMmPerHour);
}

@immutable
final class Nowcast {
  const Nowcast({
    required this.location,
    required this.issuedAt,
    required this.points,
  });

  final LatLng location;
  final DateTime issuedAt;
  final List<NowcastPoint> points;

  /// True if any timestep in the horizon shows precipitation.
  bool get hasRain =>
      points.any((p) => p.level != PrecipitationLevel.none);

  /// First timestep with rain, or `null` if dry through the whole horizon.
  NowcastPoint? get firstRainPoint {
    for (final p in points) {
      if (p.level != PrecipitationLevel.none) return p;
    }
    return null;
  }

  /// Minutes until the first rain timestep, or `null` if dry.
  int? get minutesUntilRain {
    final first = firstRainPoint;
    if (first == null) return null;
    final delta = first.timestamp.difference(issuedAt).inMinutes;
    return delta < 0 ? 0 : delta;
  }
}
