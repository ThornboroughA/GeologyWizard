import { useMemo } from "react";

interface TimelineScrubberProps {
  value: number;
  min: number;
  max: number;
  step: number;
  requestedTime: number;
  displayedTime: number | null;
  frameSource: "cached_nearest" | "exact";
  scrubState: "idle" | "dragging" | "settling";
  onChange: (value: number) => void;
  onScrubStart?: () => void;
  onScrubEnd?: () => void;
}

export function TimelineScrubber({
  value,
  min,
  max,
  step,
  requestedTime,
  displayedTime,
  frameSource,
  scrubState,
  onChange,
  onScrubStart,
  onScrubEnd
}: TimelineScrubberProps) {
  const percent = useMemo(() => ((max - value) / (max - min)) * 100, [max, min, value]);

  return (
    <section className="timeline-card">
      <div className="timeline-header">
        <h2>Geologic Timeline</h2>
        <span>
          req {requestedTime} Ma | shown {displayedTime ?? "-"} Ma
        </span>
      </div>
      <input
        className="timeline-input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onPointerDown={onScrubStart}
        onPointerUp={onScrubEnd}
        onTouchStart={onScrubStart}
        onTouchEnd={onScrubEnd}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label="Geologic timeline scrubber"
      />
      <div className="timeline-meta">
        <span>Direction: {max} Ma {"->"} {min} Ma</span>
        <span>Source: {frameSource}</span>
        <span>{scrubState === "settling" ? "Settling exact frame..." : `State: ${scrubState}`}</span>
      </div>
      <div className="timeline-scale" style={{ "--progress": `${percent}%` } as React.CSSProperties}>
        <span>{max} Ma</span>
        <span>{Math.floor((max + min) / 2)} Ma</span>
        <span>{min} Ma</span>
      </div>
    </section>
  );
}
