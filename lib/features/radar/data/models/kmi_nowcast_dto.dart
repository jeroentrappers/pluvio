import 'package:meta/meta.dart';

/// Wire-format DTO for the KMI app-viewer "forecast nowcast" response.
///
/// The endpoint is unofficial (it powers meteo.be's mobile site) so the schema
/// is fragile. We model only the fields we use and tolerate missing/null ones.
@immutable
final class KmiNowcastDto {
  const KmiNowcastDto({
    required this.issuedAt,
    required this.intervalMinutes,
    required this.precipitationMmPerHour,
  });

  /// `dateForecast` from the response, parsed to UTC.
  final DateTime issuedAt;

  /// Step between consecutive entries — 5 minutes for the standard nowcast.
  final int intervalMinutes;

  /// Precipitation values in mm/h, oldest → newest, indexed from [issuedAt].
  final List<double> precipitationMmPerHour;

  factory KmiNowcastDto.fromJson(Map<String, dynamic> json) {
    final issuedRaw = json['dateForecast'] ?? json['date_forecast'];
    if (issuedRaw is! String) {
      throw const FormatException('Missing dateForecast');
    }

    final issuedAt = DateTime.parse(issuedRaw).toUtc();

    final dataNode = json['nowcast'] ?? json['rainForecast'];
    if (dataNode is! List) {
      throw const FormatException('Missing nowcast payload');
    }

    final values = <double>[];
    for (final entry in dataNode) {
      final v = switch (entry) {
        num n => n.toDouble(),
        Map<String, dynamic> m when m['value'] is num => (m['value'] as num).toDouble(),
        Map<String, dynamic> m when m['intensity'] is num => (m['intensity'] as num).toDouble(),
        _ => null,
      };
      if (v != null) values.add(v);
    }

    final interval = switch (json['intervalMinutes']) {
      num n => n.toInt(),
      _ => 5,
    };

    return KmiNowcastDto(
      issuedAt: issuedAt,
      intervalMinutes: interval,
      precipitationMmPerHour: values,
    );
  }
}
