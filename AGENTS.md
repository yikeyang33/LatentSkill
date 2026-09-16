# Repository agent instructions

## Remote long-running jobs

- Always launch long-running downloads, training jobs, and evaluations on a
  remote server inside a named `tmux` session.
- Do not rely on a foreground SSH process or bare `nohup` for remote jobs.
- Redirect durable stdout/stderr logs to a file under the project directory.
- Immediately after launch, verify both the `tmux` session and its child
  process are alive, then report the session name, log path, and attach command.
- Before starting a replacement job, check for an existing matching process or
  tmux session to avoid duplicate downloads or evaluations.
