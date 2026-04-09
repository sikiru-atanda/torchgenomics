// Translation unit anchor for tgw::sample_gig templates. The implementation is
// header-only; this .cpp exists so the build system has a non-empty source for
// the gig sampler component (and to keep linker behaviour consistent).

#include "gig_sampler.hpp"

namespace tgw {
// Force template instantiation for std::mt19937_64 so debug builds don't
// inline-only the symbols.
template double sample_gig<std::mt19937_64>(double, double, double, std::mt19937_64&);
}  // namespace tgw
