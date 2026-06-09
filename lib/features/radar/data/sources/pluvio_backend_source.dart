import 'package:dio/dio.dart';

import '../../../../core/networking/api_failure.dart';
import '../../../../core/result/result.dart';
import '../models/pluvio_forecast_dto.dart';

/// HTTP wrapper around the Pluvio backend `/v1/forecast` endpoint.
class PluvioBackendSource {
  PluvioBackendSource({required this.dio, required this.baseUrl});

  final Dio dio;
  final String baseUrl;

  Future<Result<PluvioForecastDto, ApiFailure>> fetchForecast({
    required double latitude,
    required double longitude,
    // 24h: pulls every band the backend has — nowcast (2h), short (2-11h),
    // medium (12-23h). Long (>24h) is also fetched when it lands. The UI
    // renders whatever frames come back.
    int horizonMin = 1440,
  }) async {
    try {
      final response = await dio.get<Map<String, dynamic>>(
        '$baseUrl/v1/forecast',
        queryParameters: {
          'lat': latitude,
          'lon': longitude,
          'horizon_min': horizonMin,
        },
      );
      final data = response.data;
      if (data == null) {
        return const Result.err(ParseFailure());
      }
      return Result.ok(PluvioForecastDto.fromJson(data));
    } on DioException catch (e) {
      return Result.err(ApiFailure.fromDio(e));
    } on FormatException catch (e, st) {
      return Result.err(ParseFailure(cause: e, stackTrace: st));
    } on TypeError catch (e, st) {
      return Result.err(ParseFailure(cause: e, stackTrace: st));
    }
  }

  /// Resolve a possibly-relative overlay URL (e.g. `/v1/overlay/...`) into an
  /// absolute one the image loader can fetch.
  String absoluteUrl(String maybeRelative) {
    if (maybeRelative.startsWith('http://') || maybeRelative.startsWith('https://')) {
      return maybeRelative;
    }
    final base = baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;
    final path = maybeRelative.startsWith('/') ? maybeRelative : '/$maybeRelative';
    return '$base$path';
  }
}
