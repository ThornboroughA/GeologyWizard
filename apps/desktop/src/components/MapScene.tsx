import { useMemo, useState } from "react";

import { _GlobeView as GlobeView, MapView } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";

import type { TimelineFrameRender } from "../types";

interface MapSceneProps {
  frame: TimelineFrameRender | null;
  mode: "2d" | "3d";
  overlay: "none" | "velocity" | "boundary_class" | "event_confidence" | "uplift" | "subsidence";
}

type ViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
};

function plateFillColor(properties: Record<string, unknown>, overlay: MapSceneProps["overlay"]): [number, number, number, number] {
  const plateId = Number(properties.plateId ?? 0);
  if (overlay === "velocity") {
    const velocity = Number(properties.velocityCmYr ?? 0);
    const normalized = Math.max(0, Math.min(1, velocity / 14));
    const r = Math.round(70 + normalized * 160);
    const g = Math.round(120 + normalized * 80);
    const b = Math.round(210 - normalized * 110);
    return [r, g, b, 185];
  }

  if (overlay === "event_confidence") {
    const continuity = Number(properties.continuityScore ?? 0);
    const normalized = Math.max(0, Math.min(1, continuity));
    const r = Math.round(200 - normalized * 70);
    const g = Math.round(110 + normalized * 100);
    const b = Math.round(60 + normalized * 40);
    return [r, g, b, 180];
  }

  const base = (plateId * 37) % 255;
  return [70 + (base % 90), 105 + ((base * 2) % 85), 120 + ((base * 3) % 80), 170];
}

function boundaryColor(properties: Record<string, unknown>, overlay: MapSceneProps["overlay"]): [number, number, number, number] {
  const boundaryType = String(properties.boundaryType ?? "transform");
  if (overlay === "uplift") {
    return boundaryType === "convergent" ? [255, 204, 120, 245] : [210, 220, 228, 110];
  }
  if (overlay === "subsidence") {
    return boundaryType === "divergent" ? [140, 255, 212, 245] : [210, 220, 228, 110];
  }

  if (boundaryType === "convergent") {
    return [255, 208, 128, 235];
  }
  if (boundaryType === "divergent") {
    return [145, 255, 220, 230];
  }
  return [246, 234, 217, 205];
}

function overlayColor(properties: Record<string, unknown>, overlay: MapSceneProps["overlay"]): [number, number, number, number] {
  const confidence = Number(properties.confidence ?? 0);
  if (overlay !== "event_confidence") {
    return [0, 0, 0, 0];
  }
  const alpha = Math.round(75 + Math.max(0, Math.min(1, confidence)) * 170);
  return [250, 234, 170, alpha];
}

export function MapScene({ frame, mode, overlay }: MapSceneProps) {
  const [viewState2d, setViewState2d] = useState<ViewState>({
    longitude: 0,
    latitude: 15,
    zoom: 0.7,
    pitch: 0,
    bearing: 0
  });
  const [viewState3d, setViewState3d] = useState<ViewState>({
    longitude: 0,
    latitude: 12,
    zoom: 0.15,
    pitch: 20,
    bearing: 0
  });

  const effectiveFrame = frame ?? {
    timeMa: 0,
    source: "cache" as const,
    nearestTimeMa: 0,
    landmassGeoJson: { type: "FeatureCollection" as const, features: [] },
    boundaryGeoJson: { type: "FeatureCollection" as const, features: [] },
    overlayGeoJson: { type: "FeatureCollection" as const, features: [] }
  };

  const layers = useMemo(
    () => [
      new GeoJsonLayer({
        id: "landmasses",
        data: effectiveFrame.landmassGeoJson,
        filled: true,
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: 1,
        getFillColor: (feature: { properties: Record<string, unknown> }) => plateFillColor(feature.properties, overlay),
        getLineColor: [232, 244, 252, 160]
      }),
      new GeoJsonLayer({
        id: "boundaries",
        data: effectiveFrame.boundaryGeoJson,
        filled: false,
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: overlay === "uplift" || overlay === "subsidence" ? 2.8 : 1.8,
        getLineColor: (feature: { properties: Record<string, unknown> }) => boundaryColor(feature.properties, overlay)
      }),
      new GeoJsonLayer({
        id: "events",
        data: effectiveFrame.overlayGeoJson,
        filled: overlay === "event_confidence",
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: overlay === "event_confidence" ? 1.6 : 0.8,
        getFillColor: (feature: { properties: Record<string, unknown> }) => overlayColor(feature.properties, overlay),
        getLineColor: (feature: { properties: Record<string, unknown> }) => overlayColor(feature.properties, overlay)
      })
    ],
    [effectiveFrame, overlay]
  );

  if (!frame) {
    return <div className="map-empty">Generate a project and scrub the timeline to view tectonic history.</div>;
  }

  const view = mode === "2d" ? new MapView({ repeat: true }) : new GlobeView({});
  const viewState = mode === "2d" ? viewState2d : viewState3d;

  return (
    <div className={`deck-shell ${mode === "3d" ? "deck-shell-globe" : ""}`}>
      <DeckGL
        views={view}
        viewState={viewState}
        controller
        onViewStateChange={({ viewState: nextViewState }) => {
          const casted = nextViewState as ViewState;
          if (mode === "2d") {
            setViewState2d(casted);
          } else {
            setViewState3d(casted);
          }
        }}
        layers={layers}
      />
    </div>
  );
}
