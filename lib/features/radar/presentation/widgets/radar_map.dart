import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../../../../core/config/env.dart';
import '../../domain/radar_animation.dart';

/// Base map (OpenStreetMap raster tiles) with a single radar PNG overlay.
/// The KMI app API serves *one* composite image per timestep — no slippy
/// tiles — so we paint it via [OverlayImageLayer] anchored to a known bbox.
class RadarMap extends StatelessWidget {
  const RadarMap({
    super.key,
    required this.center,
    required this.animation,
    required this.currentIndex,
  });

  final LatLng center;
  final RadarAnimation? animation;
  final int currentIndex;

  LatLngBounds get _radarBounds => LatLngBounds(
        LatLng(Env.radarBoundsSouth, Env.radarBoundsWest),
        LatLng(Env.radarBoundsNorth, Env.radarBoundsEast),
      );

  @override
  Widget build(BuildContext context) {
    final frame = (animation == null || animation!.isEmpty)
        ? null
        : animation!.frames[currentIndex];

    return FlutterMap(
      options: MapOptions(
        initialCenter: center,
        initialZoom: 7.5,
        minZoom: 5,
        maxZoom: 11,
        interactionOptions: const InteractionOptions(
          flags: InteractiveFlag.pinchZoom |
              InteractiveFlag.drag |
              InteractiveFlag.doubleTapZoom,
        ),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'app.appmire.pluvio',
          maxNativeZoom: 19,
        ),
        if (frame != null)
          OverlayImageLayer(
            overlayImages: [
              OverlayImage(
                bounds: _radarBounds,
                imageProvider: NetworkImage(frame.imageUrl),
                opacity: 0.85,
                gaplessPlayback: true,
              ),
            ],
          ),
        MarkerLayer(
          markers: [
            Marker(
              point: center,
              width: 14,
              height: 14,
              child: _UserDot(scheme: Theme.of(context).colorScheme),
            ),
          ],
        ),
        const _AttributionBox(),
      ],
    );
  }
}

class _UserDot extends StatelessWidget {
  const _UserDot({required this.scheme});

  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: scheme.primary,
        border: Border.all(color: Colors.white, width: 2),
        shape: BoxShape.circle,
      ),
    );
  }
}

class _AttributionBox extends StatelessWidget {
  const _AttributionBox();

  @override
  Widget build(BuildContext context) {
    return const RichAttributionWidget(
      attributions: [
        TextSourceAttribution('© OpenStreetMap contributors'),
        TextSourceAttribution('Radar © KMI / IRM (app.meteo.be)'),
      ],
    );
  }
}
