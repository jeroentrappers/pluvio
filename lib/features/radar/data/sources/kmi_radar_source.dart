import 'package:dio/dio.dart';

import '../../../../core/networking/api_failure.dart';
import '../../../../core/result/result.dart';
import '../models/kmi_radar_capabilities_dto.dart';

/// Talks to the KMI WMS endpoint. Two responsibilities only:
///   1. discover which radar frames are available (GetCapabilities)
///   2. expose the URL template needed by [flutter_map] to render each one
class KmiRadarSource {
  KmiRadarSource({
    required this.dio,
    required this.wmsUrl,
    required this.layer,
  });

  final Dio dio;
  final String wmsUrl;
  final String layer;

  Future<Result<KmiRadarCapabilitiesDto, ApiFailure>> fetchCapabilities() async {
    try {
      final response = await dio.get<String>(
        wmsUrl,
        queryParameters: {
          'service': 'WMS',
          'request': 'GetCapabilities',
          'version': '1.3.0',
        },
        options: Options(responseType: ResponseType.plain),
      );
      final body = response.data;
      if (body == null || body.isEmpty) {
        return const Result.err(ParseFailure());
      }
      final timeDim = _extractTimeDimension(body, layer);
      if (timeDim == null) {
        return const Result.err(ParseFailure());
      }
      return Result.ok(KmiRadarCapabilitiesDto.fromTimeDimension(timeDim));
    } on DioException catch (e) {
      return Result.err(ApiFailure.fromDio(e));
    }
  }

  /// Builds a per-frame WMS GetMap URL template suitable for [TileLayer.wmsOptions].
  Uri tileTemplateForFrame(DateTime time) {
    return Uri.parse(wmsUrl).replace(
      queryParameters: {
        'service': 'WMS',
        'version': '1.3.0',
        'request': 'GetMap',
        'layers': layer,
        'styles': '',
        'format': 'image/png',
        'transparent': 'true',
        'time': time.toUtc().toIso8601String(),
        'crs': 'EPSG:3857',
      },
    );
  }

  /// Pull `<Dimension name="time">` for a specific layer out of the raw XML.
  /// Deliberately string-based; full XML parsing would pull in a dep we don't
  /// otherwise need and is overkill for one element.
  String? _extractTimeDimension(String xml, String layerName) {
    final layerPattern = RegExp(
      r'<Layer[^>]*>([\s\S]*?</Layer>)',
      multiLine: true,
    );
    for (final m in layerPattern.allMatches(xml)) {
      final block = m.group(1) ?? '';
      if (!block.contains('<Name>$layerName</Name>')) continue;
      final dim = RegExp(
        r'<Dimension[^>]*name="time"[^>]*>([\s\S]*?)</Dimension>',
        multiLine: true,
      ).firstMatch(block);
      if (dim != null) return dim.group(1)?.trim();
    }
    return null;
  }
}
