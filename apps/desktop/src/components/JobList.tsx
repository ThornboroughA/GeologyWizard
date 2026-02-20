import type { JobSummary } from "../types";

interface JobListProps {
  jobs: JobSummary[];
}

export function JobList({ jobs }: JobListProps) {
  if (jobs.length === 0) {
    return <p className="muted">No jobs yet.</p>;
  }

  return (
    <ul className="job-list">
      {jobs.map((job) => (
        <li key={job.jobId} className="job-item">
          <header>
            <strong>{job.kind}</strong>
            <span className={`status ${job.status}`}>{job.status}</span>
          </header>
          <p>{job.message}</p>
          <progress max={1} value={job.progress} />
          {job.artifacts.length > 0 ? <small>{job.artifacts.length} artifacts ready.</small> : null}
          {job.error ? <small className="error">{job.error}</small> : null}
        </li>
      ))}
    </ul>
  );
}
