import 'package:meta/meta.dart';

/// Wire-format DTO for `https://app.meteo.be/services/appv4/?s=getForecasts`.
///
/// We deliberately model only the fields Pluvio uses today (city, animation).
/// The endpoint is unofficial; missing/unexpected fields are tolerated so a
/// schema drift doesn't blow up the whole parse.
@immutable
final class KmiGetForecastsDto {
  const KmiGetForecastsDto({
    required this.cityName,
    required this.country,
    required this.animation,
  });

  final String? cityName;
  final String? country;
  final KmiAnimationDto animation;

  factory KmiGetForecastsDto.fromJson(Map<String, dynamic> json) {
    final animationNode = json['animation'];
    if (animationNode is! Map<String, dynamic>) {
      throw const FormatException('Missing animation block');
    }
    return KmiGetForecastsDto(
      cityName: json['cityName'] as String?,
      country: json['country'] as String?,
      animation: KmiAnimationDto.fromJson(animationNode),
    );
  }
}

@immutable
final class KmiAnimationDto {
  const KmiAnimationDto({
    required this.type,
    required this.sequence,
  });

  /// Interval label from the wire, e.g. `"10min"`. Maps to a [Duration] via
  /// [interval]; unknown values fall back to 10 minutes.
  final String? type;
  final List<KmiAnimationFrameDto> sequence;

  Duration get interval {
    return switch (type) {
      '5min' => const Duration(minutes: 5),
      '10min' => const Duration(minutes: 10),
      '15min' => const Duration(minutes: 15),
      _ => const Duration(minutes: 10),
    };
  }

  factory KmiAnimationDto.fromJson(Map<String, dynamic> json) {
    final seq = json['sequence'];
    if (seq is! List) {
      throw const FormatException('Missing animation sequence');
    }
    return KmiAnimationDto(
      type: json['type'] as String?,
      sequence: [
        for (final entry in seq)
          if (entry is Map<String, dynamic>) KmiAnimationFrameDto.fromJson(entry),
      ],
    );
  }
}

@immutable
final class KmiAnimationFrameDto {
  const KmiAnimationFrameDto({
    required this.time,
    required this.imageUrl,
    required this.valueMmPer10Min,
  });

  /// Frame timestamp parsed from the wire (with timezone offset preserved →
  /// always converted to UTC here so downstream code can compare safely).
  final DateTime time;

  /// Pre-signed CDN PNG URL.
  final String imageUrl;

  /// Precipitation rate at the user's location for this frame, in mm/10min.
  final double valueMmPer10Min;

  factory KmiAnimationFrameDto.fromJson(Map<String, dynamic> json) {
    final timeStr = json['time'];
    final uri = json['uri'];
    final value = json['value'];
    if (timeStr is! String || uri is! String) {
      throw const FormatException('Missing time/uri on animation frame');
    }
    return KmiAnimationFrameDto(
      time: DateTime.parse(timeStr).toUtc(),
      imageUrl: uri,
      valueMmPer10Min: switch (value) {
        num n => n.toDouble(),
        _ => 0,
      },
    );
  }
}
