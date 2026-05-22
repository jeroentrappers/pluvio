// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for French (`fr`).
class AppLocalizationsFr extends AppLocalizations {
  AppLocalizationsFr([String locale = 'fr']) : super(locale);

  @override
  String get appTitle => 'Pluvio';

  @override
  String get refresh => 'Actualiser';

  @override
  String get locationError => 'Impossible de déterminer votre position.';

  @override
  String get radarError => 'Impossible de récupérer les données radar.';

  @override
  String get nowcastError => 'Impossible de récupérer la prévision immédiate.';

  @override
  String get nowcastDry => 'Pas de pluie prévue dans les deux prochaines heures.';

  @override
  String get nowcastRaining => 'Il pleut actuellement.';

  @override
  String nowcastRainInMinutes(int minutes) {
    final intl.NumberFormat minutesNumberFormat = intl.NumberFormat.decimalPattern(localeName);
    final String minutesString = minutesNumberFormat.format(minutes);

    return 'Pluie prévue dans $minutesString min.';
  }
}
