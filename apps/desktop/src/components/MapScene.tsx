import type { TimelineFrame } from "../types";

interface MapSceneProps {
  frame: TimelineFrame | null;
  mode: "2d" | "3d";
  overlay: "none" | "velocity" | "boundary_class" | "event_confidence" | "uplift" | "subsidence";
}

function lonToX(lon: number, width: number): number {
  return ((lon + 180) / 360) * width;
}

function latToY(lat: number, height: number): number {
  return ((90 - lat) / 180) * height;
}

function polygonPath(coordinates: number[][], width: number, height: number): string {
  return coordinates
    .map((point, index) => {
      const x = lonToX(point[0], width);
      const y = latToY(point[1], height);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function plateFill(frame: TimelineFrame, plateId: number, overlay: MapSceneProps["overlay"]): string {
  if (overlay === "velocity") {
    const kin = frame.plateKinematics.find((item) => item.plateId === plateId);
    const velocity = kin?.velocityCmYr ?? 0;
    const normalized = Math.min(1, velocity / 14);
    const hue = 210 - normalized * 140;
    return `hsl(${hue} 68% 52% / 0.6)`;
  }
  if (overlay === "event_confidence") {
    const confidence = frame.eventOverlays.reduce((acc, event) => acc + event.confidence, 0) / Math.max(1, frame.eventOverlays.length);
    const hue = 35 + confidence * 80;
    return `hsl(${hue} 58% 46% / 0.56)`;
  }
  return `hsl(${(plateId * 35) % 360} 38% 42% / 0.55)`;
}

function boundaryClass(boundaryType: string, overlay: MapSceneProps["overlay"]): string {
  if (overlay === "uplift") {
    return boundaryType === "convergent" ? "uplift-focus" : "muted-boundary";
  }
  if (overlay === "subsidence") {
    return boundaryType === "divergent" ? "subsidence-focus" : "muted-boundary";
  }
  return boundaryType;
}

export function MapScene({ frame, mode, overlay }: MapSceneProps) {
  if (!frame) {
    return <div className="map-empty">Generate a project and scrub the timeline to view tectonic history.</div>;
  }

  if (mode === "3d") {
    return (
      <div className="globe-shell">
        <div className="globe">
          {frame.plateGeometries.slice(0, 24).map((plate, index) => {
            const center = plate.geometry.coordinates[0][0];
            const x = 50 + (center[0] / 180) * 32;
            const y = 50 - (center[1] / 90) * 32;
            const kin = frame.plateKinematics.find((item) => item.plateId === plate.plateId);
            const dotScale = overlay === "velocity" ? Math.max(6, (kin?.velocityCmYr ?? 1) * 1.2) : 8;
            return (
              <span
                key={plate.plateId}
                className="globe-dot"
                style={{
                  left: `${x}%`,
                  top: `${y}%`,
                  width: `${dotScale}px`,
                  height: `${dotScale}px`,
                  animationDelay: `${index * 50}ms`
                }}
              />
            );
          })}
        </div>
      </div>
    );
  }

  const width = 960;
  const height = 480;

  return (
    <svg className="map-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="2D tectonic map view">
      <defs>
        <linearGradient id="ocean" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#0e3c52" />
          <stop offset="100%" stopColor="#123047" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill="url(#ocean)" rx="14" />
      {frame.plateGeometries.map((plate) => (
        <path
          key={plate.plateId}
          d={polygonPath(plate.geometry.coordinates[0], width, height)}
          className="plate-path"
          style={{ fill: plateFill(frame, plate.plateId, overlay) }}
        />
      ))}
      {frame.boundaryGeometries.map((boundary) => (
        <line
          key={boundary.segmentId}
          x1={lonToX(boundary.geometry.coordinates[0][0], width)}
          y1={latToY(boundary.geometry.coordinates[0][1], height)}
          x2={lonToX(boundary.geometry.coordinates[1][0], width)}
          y2={latToY(boundary.geometry.coordinates[1][1], height)}
          className={`boundary ${boundaryClass(boundary.boundaryType, overlay)}`}
        />
      ))}
      {overlay === "event_confidence"
        ? frame.eventOverlays.map((event) => {
            const geometry = event.regionGeometry;
            const coordinates = geometry.type === "LineString" ? geometry.coordinates : geometry.coordinates[0];
            if (coordinates.length === 0) {
              return null;
            }
            const center = coordinates[0];
            const radius = 2 + event.confidence * 8;
            return (
              <circle
                key={event.eventId}
                cx={lonToX(center[0], width)}
                cy={latToY(center[1], height)}
                r={radius}
                fill={`rgba(255, 236, 173, ${0.2 + event.confidence * 0.5})`}
                stroke="rgba(255, 250, 220, 0.8)"
                strokeWidth={0.8}
              />
            );
          })
        : null}
    </svg>
  );
}
