"""
Global configuration for XLB.

Call :func:`init` once at the start of every script to select the velocity
set, compute backend, and precision policy.  All operators read their
defaults from :class:`DefaultConfig` when explicit arguments are omitted.
"""

import os

from xlb.compute_backend import ComputeBackend
from dataclasses import dataclass
from xlb.precision_policy import PrecisionPolicy


@dataclass
class DefaultConfig:
    """Singleton holding the active global configuration.

    Attributes are set by :func:`init` and read by operators, grids, and
    helpers throughout XLB.

    Attributes
    ----------
    default_precision_policy : PrecisionPolicy or None
        Active precision policy (compute / store dtype pair).
    velocity_set : VelocitySet or None
        Active lattice velocity set.
    default_backend : ComputeBackend or None
        Active compute backend.
    """

    default_precision_policy = None
    velocity_set = None
    default_backend = None


def _warp_init_and_select_cuda_device():
    """Initialize Warp and pin the default CUDA device for single-GPU XLB runs.

    With multiple GPUs, Warp's default device for allocations and launches can
    otherwise diverge. Set ``XLB_WARP_DEVICE`` (e.g. ``cuda:0`` or ``cuda:1``)
    to choose which GPU Warp uses; defaults to ``cuda:0`` when unset.
    """
    import warp as wp

    wp.init()  # TODO: Must be removed in the future versions of WARP
    if wp.get_cuda_device_count() == 0:
        return
    choice = os.environ.get("XLB_WARP_DEVICE", "cuda:0").strip()
    try:
        wp.set_device(choice)
    except Exception:
        try:
            wp.set_device("cuda:0")
        except Exception:
            pass


def _maybe_enable_fast_math():
    """Force ``fast_math`` on for every Warp module when ``XLB_FAST_MATH`` is set.

    Warp offers no global switch for this.  ``warp.config`` has no ``fast_math``
    field at all, and :class:`warp.context.Module` hardcodes
    ``options["fast_math"] = False`` when it builds its option dict, so setting
    ``wp.config.fast_math = True`` silently does nothing.  The documented
    alternative, :func:`warp.set_module_options`, targets a *Python* module --
    but Neon compiles every container kernel into its own anonymous Warp module
    via ``@wp.kernel(module="unique")`` (see ``neon/container.py``), so there is
    no importable module to point it at.

    Patching ``Module.__init__`` is therefore the only hook that reaches both
    XLB's own operator modules and Neon's per-container ones.

    Enabling this maps fp32 division and square root onto the fast hardware
    intrinsics and flushes denormals to zero.  On the D3Q27 KBC+Smagorinsky
    collision path this measured 2488 -> 1824 SASS instructions (-27%) and
    removed all 34 division range checks, at a cost of ~11 registers
    (117 -> 128, i.e. 17 -> 16 warps/SM on sm_86).

    It is a real numerical change, so it is opt-in and off by default.  Validate
    against a reference case, not just MLUPS.

    Returns
    -------
    bool
        ``True`` if fast math was enabled.
    """
    if os.environ.get("XLB_FAST_MATH", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return False

    from warp.context import Module

    if not getattr(Module, "_xlb_fast_math_patched", False):
        _original_init = Module.__init__

        def _init_with_fast_math(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            self.options["fast_math"] = True

        Module.__init__ = _init_with_fast_math
        Module._xlb_fast_math_patched = True

    print("XLB: fast_math ENABLED for all Warp modules (XLB_FAST_MATH is set)")
    return True


def init(velocity_set, default_backend, default_precision_policy):
    """Initialize the global XLB configuration.

    Must be called before creating any grid, operator, or field.

    Parameters
    ----------
    velocity_set : VelocitySet
        Lattice velocity set (e.g. ``D3Q19``).
    default_backend : ComputeBackend
        Compute backend to use (JAX, WARP, or NEON).
    default_precision_policy : PrecisionPolicy
        Precision policy for compute and storage dtypes.
    """
    DefaultConfig.velocity_set = velocity_set
    DefaultConfig.default_backend = default_backend
    DefaultConfig.default_precision_policy = default_precision_policy

    if default_backend == ComputeBackend.WARP:
        _warp_init_and_select_cuda_device()
        _maybe_enable_fast_math()
    elif default_backend == ComputeBackend.NEON:
        import warp as wp
        import neon

        # wp.config.mode = "release"
        # wp.config.llvm_cuda = False
        # wp.config.verbose = True
        # wp.verbose_warnings = True

        _warp_init_and_select_cuda_device()

        # Must precede any kernel definition: it changes the codegen options that
        # every Warp module is constructed with.
        _maybe_enable_fast_math()

        # It's a good idea to always clear the kernel cache when developing new native or codegen features
        wp.build.clear_kernel_cache()

        # !!! DO THIS BEFORE DEFINING/USING ANY KERNELS WITH CUSTOM TYPES
        neon.init()

    elif default_backend == ComputeBackend.JAX:
        check_backend_support()
    else:
        raise ValueError(f"Unsupported compute backend: {default_backend}")


def default_backend() -> ComputeBackend:
    """Return the currently configured compute backend."""
    return DefaultConfig.default_backend


def check_backend_support():
    """Print a summary of available JAX hardware accelerators."""
    import jax

    if jax.devices()[0].platform == "gpu":
        gpus = jax.devices("gpu")
        if len(gpus) > 1:
            print("Multi-GPU support is available: {} GPUs detected.".format(len(gpus)))
        elif len(gpus) == 1:
            print("Single-GPU support is available: 1 GPU detected.")

    elif jax.devices()[0].platform == "tpu":
        tpus = jax.devices("tpu")
        if len(tpus) > 1:
            print("Multi-TPU support is available: {} TPUs detected.".format(len(tpus)))
        elif len(tpus) == 1:
            print("Single-TPU support is available: 1 TPU detected.")
    else:
        print("No GPU support is available; CPU fallback will be used.")
