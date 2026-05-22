import 'package:dio/dio.dart';

import '../../../../core/networking/api_failure.dart';
import '../../../../core/result/result.dart';
import '../models/kmi_nowcast_dto.dart';

/// Thin HTTP wrapper around KMI's per-location nowcast endpoint. Knows how to
/// build the request and translate Dio errors to [ApiFailure] — nothing else.
class KmiNowcastSource {
  KmiNowcastSource({required this.dio, required this.baseUrl});

  final Dio dio;
  final String baseUrl;

  Future<Result<KmiNowcastDto, ApiFailure>> fetch({
    required double latitude,
    required double longitude,
  }) async {
    try {
      final response = await dio.get<Map<String, dynamic>>(
        '$baseUrl/forecasts',
        queryParameters: {
          'lat': latitude.toStringAsFixed(5),
          'lon': longitude.toStringAsFixed(5),
          'view': 'nowcast',
        },
      );
      final data = response.data;
      if (data == null) {
        return const Result.err(ParseFailure());
      }
      return Result.ok(KmiNowcastDto.fromJson(data));
    } on DioException catch (e) {
      return Result.err(ApiFailure.fromDio(e));
    } on FormatException catch (e, st) {
      return Result.err(ParseFailure(cause: e, stackTrace: st));
    } on TypeError catch (e, st) {
      return Result.err(ParseFailure(cause: e, stackTrace: st));
    }
  }
}
