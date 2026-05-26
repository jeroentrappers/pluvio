import 'dart:convert';

import 'package:crypto/crypto.dart';

/// The KMI mobile-app API (`app.meteo.be/services/appv4`) signs every
/// request with an `md5("<salt>;<method>;<DD/MM/YYYY>")` key. The salt is
/// the same one used by the upstream Apache-2.0 `irm-kmi-api` Python package
/// (reverse-engineered from the official KMI mobile app). The hash rotates
/// daily, so signing always uses the device's *local* date.
abstract final class KmiApiSigning {
  static const String _salt = 'r9EnW374jkJ9acc';

  /// [method] is the value of the `s=` query parameter on the upstream call
  /// (e.g. `getForecasts`). [clock] is injectable so tests are deterministic.
  static String key(String method, {DateTime Function()? clock}) {
    final now = (clock ?? DateTime.now).call();
    final ymd = '${_pad2(now.day)}/${_pad2(now.month)}/${now.year}';
    final raw = '$_salt;$method;$ymd';
    return md5.convert(utf8.encode(raw)).toString();
  }

  static String _pad2(int n) => n.toString().padLeft(2, '0');
}
