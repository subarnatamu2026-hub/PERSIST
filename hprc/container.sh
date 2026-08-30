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
      # Read-only image (no -w). Don't bind host $HOME (--no-home); bind /scratch
      # (mount point baked into the image) and a writable /tmp from scratch.
      # HOME/TMPDIR/caches point at scratch so writes land on the bound fs.
      ch-run --no-home -b /scratch -b "$SCRATCH/tmp:/tmp" \
        "${CH_IMAGE:-$SCRATCH/craftium.sqfs}" -- \
        bash -lc "export HOME='$SCRATCH' TMPDIR='$SCRATCH/tmp' XDG_CACHE_HOME='$SCRATCH/.cache'; $cmd"
      ;;
    *)
      "${CONTAINER:-singularity}" exec --bind /scratch "${SIF:-$SCRATCH/craftium.sif}" \
        bash -lc "$cmd"
      ;;
  esac
}
