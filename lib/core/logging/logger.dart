import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:talker_flutter/talker_flutter.dart';
import 'package:talker_riverpod_logger/talker_riverpod_logger.dart';

/// Project-wide logger. Wraps Talker so we have a single seam for swapping
/// the implementation (e.g. to also forward to Sentry in production).
abstract final class AppLogger {
  static final Talker _talker = TalkerFlutter.init(
    settings: TalkerSettings(
      enabled: true,
      useConsoleLogs: kDebugMode,
      maxHistoryItems: 500,
    ),
  );

  static Talker get talker => _talker;

  static void init() {
    _talker.info('Pluvio logger initialized');
  }

  static void captureFlutterError(FlutterErrorDetails details) {
    _talker.handle(details.exception, details.stack, details.context?.toString());
  }

  static void captureException(Object error, StackTrace stack, [String? context]) {
    _talker.handle(error, stack, context);
  }

  static ProviderObserver riverpodObserver() {
    return TalkerRiverpodObserver(
      talker: _talker,
      settings: const TalkerRiverpodLoggerSettings(
        printProviderAdded: false,
        printProviderUpdated: false,
        printProviderDisposed: false,
        printProviderFailed: true,
      ),
    );
  }
}
