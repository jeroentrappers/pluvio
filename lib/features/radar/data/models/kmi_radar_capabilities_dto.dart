import 'package:meta/meta.dart';

/// Parsed subset of a WMS GetCapabilities response — we only need the list of
/// `<Dimension name="time">` values for our chosen radar layer.
@immutable
final class KmiRadarCapabilitiesDto {
  const KmiRadarCapabilitiesDto({required this.timeSteps});

  final List<DateTime> timeSteps;

  /// Parses the comma-separated `<Dimension name="time">` payload exposed by
  /// WMS endpoints. Each value is an ISO-8601 timestamp; some servers return
  /// ranges (`start/end/period`) which we expand naively.
  factory KmiRadarCapabilitiesDto.fromTimeDimension(String raw) {
    final results = <DateTime>[];
    for (final token in raw.split(',')) {
      final trimmed = token.trim();
      if (trimmed.isEmpty) continue;
      if (trimmed.contains('/')) {
        final parts = trimmed.split('/');
        if (parts.length == 3) {
          final start = DateTime.parse(parts[0]).toUtc();
          final end = DateTime.parse(parts[1]).toUtc();
          final period = _parseIso8601Duration(parts[2]);
          if (period != null && period.inSeconds > 0) {
            for (var t = start; !t.isAfter(end); t = t.add(period)) {
              results.add(t);
            }
            continue;
          }
        }
      }
      try {
        results.add(DateTime.parse(trimmed).toUtc());
      } on FormatException {
        // Skip unparseable token, keep going.
      }
    }
    results.sort();
    return KmiRadarCapabilitiesDto(timeSteps: results);
  }
}

/// Tiny PT5M / PT1H / P1D parser — only the forms we expect from WMS servers.
Duration? _parseIso8601Duration(String s) {
  final m = RegExp(r'^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$').firstMatch(s);
  if (m == null) return null;
  final d = int.tryParse(m.group(1) ?? '0') ?? 0;
  final h = int.tryParse(m.group(2) ?? '0') ?? 0;
  final mi = int.tryParse(m.group(3) ?? '0') ?? 0;
  final sec = int.tryParse(m.group(4) ?? '0') ?? 0;
  return Duration(days: d, hours: h, minutes: mi, seconds: sec);
}
