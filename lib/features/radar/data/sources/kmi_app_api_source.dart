import 'package:dio/dio.dart';

import '../../../../core/networking/api_failure.dart';
import '../../../../core/result/result.dart';
import '../api_signing.dart';
import '../models/kmi_get_forecasts_dto.dart';

/// Thin HTTP wrapper around the KMI mobile-app `getForecasts` endpoint. Knows
/// how to sign + build the request and translate Dio errors to [ApiFailure].
class KmiAppApiSource {
  KmiAppApiSource({
    required this.dio,
    required this.baseUrl,
    this.signingClock,
  });

  final Dio dio;
  final String baseUrl;
  final DateTime Function()? signingClock;

  static const _method = 'getForecasts';

  Future<Result<KmiGetForecastsDto, ApiFailure>> fetch({
    required double latitude,
    required double longitude,
  }) async {
    try {
      final response = await dio.get<Map<String, dynamic>>(
        baseUrl,
        queryParameters: {
          's': _method,
          'k': KmiApiSigning.key(_method, clock: signingClock),
          'lat': _round(latitude),
          'long': _round(longitude),
        },
      );
      final data = response.data;
      if (data == null) {
        return const Result.err(ParseFailure());
      }
      return Result.ok(KmiGetForecastsDto.fromJson(data));
    } on DioException catch (e) {
      return Result.err(ApiFailure.fromDio(e));
    } on FormatException catch (e, st) {
      return Result.err(ParseFailure(cause: e, stackTrace: st));
    } on TypeError catch (e, st) {
      return Result.err(ParseFailure(cause: e, stackTrace: st));
    }
  }

  /// KMI is sensitive to over-precise coordinates — it returns inconsistent
  /// city names when the precision drifts. Six decimals matches the upstream
  /// `irm-kmi-api` rounding.
  static double _round(double v) => double.parse(v.toStringAsFixed(6));
}
