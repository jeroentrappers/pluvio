import 'package:flutter/material.dart';
import 'package:flutter_hooks/flutter_hooks.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../../l10n/app_localizations.dart';
import '../../location/application/location_providers.dart';
import '../application/radar_providers.dart';
import '../domain/radar_animation.dart';
import 'widgets/precipitation_legend.dart';
import 'widgets/radar_map.dart';
import 'widgets/timeline_slider.dart';

class RadarScreen extends HookConsumerWidget {
  const RadarScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final location = ref.watch(currentLocationProvider);
    final frameIndex = useState<int?>(null);

    return Scaffold(
      appBar: AppBar(
        title: Text(l10n.appTitle),
        actions: [
          IconButton(
            tooltip: l10n.refresh,
            icon: const Icon(Icons.refresh),
            onPressed: () {
              final loc = location.value;
              if (loc != null) ref.invalidate(radarAnimationProvider(loc));
            },
          ),
        ],
      ),
      body: location.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (_, __) => _ErrorView(message: l10n.locationError),
        data: (latLng) => _RadarBody(
          location: latLng,
          frameIndex: frameIndex,
        ),
      ),
    );
  }
}

class _RadarBody extends ConsumerWidget {
  const _RadarBody({required this.location, required this.frameIndex});

  final LatLng location;
  final ValueNotifier<int?> frameIndex;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final state = ref.watch(radarAnimationProvider(location));

    return state.when(
      loading: () => _MapShell(
        center: location,
        animation: null,
        frameIndex: 0,
        onIndexChanged: (_) {},
        body: const _LoadingHint(),
      ),
      error: (_, __) => _MapShell(
        center: location,
        animation: null,
        frameIndex: 0,
        onIndexChanged: (_) {},
        body: _ErrorView(message: l10n.radarError),
      ),
      data: (result) => result.when(
        ok: (anim) {
          final idx = frameIndex.value ?? anim.currentIndex;
          return _MapShell(
            center: location,
            animation: anim,
            frameIndex: idx,
            onIndexChanged: (v) => frameIndex.value = v,
            body: _NowcastSummary(animation: anim),
          );
        },
        err: (_) => _MapShell(
          center: location,
          animation: null,
          frameIndex: 0,
          onIndexChanged: (_) {},
          body: _ErrorView(message: l10n.radarError),
        ),
      ),
    );
  }
}

class _MapShell extends StatelessWidget {
  const _MapShell({
    required this.center,
    required this.animation,
    required this.frameIndex,
    required this.onIndexChanged,
    required this.body,
  });

  final LatLng center;
  final RadarAnimation? animation;
  final int frameIndex;
  final ValueChanged<int> onIndexChanged;
  final Widget body;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          flex: 3,
          child: RadarMap(
            center: center,
            animation: animation,
            currentIndex: frameIndex,
          ),
        ),
        if (animation != null && !animation!.isEmpty)
          TimelineSlider(
            animation: animation!,
            currentIndex: frameIndex,
            onChanged: onIndexChanged,
          ),
        Expanded(
          flex: 2,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: body,
          ),
        ),
      ],
    );
  }
}

class _NowcastSummary extends StatelessWidget {
  const _NowcastSummary({required this.animation});

  final RadarAnimation animation;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final headline = _headline(l10n, animation);
    final futureFrames =
        animation.frames.where((f) => !f.timestamp.isBefore(animation.referenceTime)).toList();

    // Flex.spacing (Flutter 3.27+) avoids the .map(Padding) trick that
    // illegally wraps an Expanded in Padding (ParentData mismatch).
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 12,
      children: [
        Text(headline, style: Theme.of(context).textTheme.titleLarge),
        Expanded(child: _Bars(frames: futureFrames)),
        const PrecipitationLegend(),
      ],
    );
  }

  String _headline(AppLocalizations l10n, RadarAnimation animation) {
    final minutes = animation.minutesUntilRain;
    if (minutes == null) return l10n.nowcastDry;
    if (minutes == 0) return l10n.nowcastRaining;
    return l10n.nowcastRainInMinutes(minutes);
  }
}

class _Bars extends StatelessWidget {
  const _Bars({required this.frames});

  final List<RadarFrame> frames;

  @override
  Widget build(BuildContext context) {
    if (frames.isEmpty) return const SizedBox.shrink();
    final scheme = Theme.of(context).colorScheme;
    final maxRate = frames
        .map((f) => f.valueMmPerHour)
        .fold<double>(0, (m, v) => v > m ? v : m)
        .clamp(0.5, double.infinity);

    return LayoutBuilder(
      builder: (_, constraints) {
        final barWidth = (constraints.maxWidth / frames.length).clamp(2.0, 16.0);
        return Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            for (final f in frames)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 1),
                child: Container(
                  width: barWidth - 2,
                  height: (f.valueMmPerHour / maxRate) * constraints.maxHeight,
                  color: PrecipitationPalette.of(f.level, scheme),
                ),
              ),
          ],
        );
      },
    );
  }
}

class _LoadingHint extends StatelessWidget {
  const _LoadingHint();

  @override
  Widget build(BuildContext context) {
    return const Center(child: CircularProgressIndicator());
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Text(message, textAlign: TextAlign.center),
      ),
    );
  }
}
