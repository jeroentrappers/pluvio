import 'package:meta/meta.dart';

/// Wire-format DTO for the Pluvio backend `GET /v1/forecast` response.
///
/// Shape (see backend/src/pluvio_backend/api.py → ForecastDto):
/// ```json
/// {
///   "issued_at": "2026-05-27T07:00:00Z",
///   "location": {"lat": 50.85, "lon": 4.35},
///   "model_version": "stub-0.1",
///   "horizon_min": 1440,
///   "frames": [
///     {"band":"nowcast","lead_min":0,"valid_time":"...",
///      "rate_mm_per_h":0.0,"overlay_url":"/v1/overlay/nowcast/0.png?t=..."}
///   ]
/// }
/// ```
@immutable
final class PluvioForecastDto {
  const PluvioForecastDto({
    required this.issuedAt,
    required this.modelVersion,
    required this.frames,
  });

  final DateTime issuedAt;
  final String modelVersion;
  final List<PluvioFrameDto> frames;

  factory PluvioForecastDto.fromJson(Map<String, dynamic> json) {
    final issuedRaw = json['issued_at'];
    final framesRaw = json['frames'];
    if (issuedRaw is! String || framesRaw is! List) {
      throw const FormatException('Missing issued_at / frames');
    }
    return PluvioForecastDto(
      issuedAt: DateTime.parse(issuedRaw).toUtc(),
      modelVersion: (json['model_version'] as String?) ?? 'unknown',
      frames: [
        for (final f in framesRaw)
          if (f is Map<String, dynamic>) PluvioFrameDto.fromJson(f),
      ],
    );
  }
}

@immutable
final class PluvioFrameDto {
  const PluvioFrameDto({
    required this.band,
    required this.leadMin,
    required this.validTime,
    required this.rateMmPerHour,
    required this.overlayUrl,
  });

  final String band;
  final int leadMin;
  final DateTime validTime;
  final double rateMmPerHour;
  final String overlayUrl;

  factory PluvioFrameDto.fromJson(Map<String, dynamic> json) {
    return PluvioFrameDto(
      band: (json['band'] as String?) ?? 'nowcast',
      leadMin: switch (json['lead_min']) {
        num n => n.toInt(),
        _ => 0,
      },
      validTime: DateTime.parse(json['valid_time'] as String).toUtc(),
      rateMmPerHour: switch (json['rate_mm_per_h']) {
        num n => n.toDouble(),
        _ => 0,
      },
      overlayUrl: (json['overlay_url'] as String?) ?? '',
    );
  }
}
