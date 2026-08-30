# Source this to get ctr_exec(): run a bash command string inside the project
# container. Supports Singularity/Apptainer (Grace) and Charliecloud (FASTER).
#
# Env knobs:
#   CONTAINER_KIND    singularity (default) | charliecloud
#   CONTAINER_MODULE  optional module to load first (e.g. charliecloud/0.33)
#   singularity:      CONTAINER (cmd, default singularity), SIF (default $SCRATCH/craftium.sif)
#   charliecloud:     CH_IMAGE (squashfs or dir, default $SCRATCH/craftium.sqfs)

ctr_load_module () { [ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE" || true; }

ctr_exec () {
  local cmd="$1"
  ctr_load_module
  case "${CONTAINER_KIND:-singularity}" in
    charliecloud|ch|ch-run)
      mkdir -p "$SCRATCH/tmp" "$SCRATCH/.cache"
      # Charliecloud keeps the host env; point HOME/TMPDIR/caches at scratch so
      # nothing tries to write to a non-existent in-container home.
      ch-run -w -b /scratch "${CH_IMAGE:-$SCRATCH/craftium.sqfs}" -- \
        bash -lc "export HOME='$SCRATCH' TMPDIR='$SCRATCH/tmp' XDG_CACHE_HOME='$SCRATCH/.cache'; $cmd"
      ;;
    *)
      "${CONTAINER:-singularity}" exec --bind /scratch "${SIF:-$SCRATCH/craftium.sif}" \
        bash -lc "$cmd"
      ;;
  esac
}
