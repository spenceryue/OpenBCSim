/*
Copyright (c) 2015, Sigurd Storve
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <memory>
#include <vector>
#include <complex>
#include "export_macros.hpp"
#include "BCSimConfig.hpp"
#include "ScanSequence.hpp"
#include "BeamProfile.hpp"

namespace bcsim {

// Interface for simulator algorithm implementations
class IAlgorithm {
public:
    typedef std::shared_ptr<IAlgorithm> s_ptr;
    typedef std::unique_ptr<IAlgorithm> u_ptr;
    
    virtual ~IAlgorithm() { }

    // Set misc. parameters. Available keys depends on the algorithm.
    virtual void set_parameter(const std::string&, const std::string& value)            = 0;
    
    // Get misc. parameters. Available keys depends on the algorithm.
    virtual std::string get_parameter(const std::string& key) const                     = 0;

    // Clear all fixed point scatterers.
    virtual void clear_fixed_scatterers()                                               = 0;

    // Add a new set of fixed point scatterers.
    virtual void add_fixed_scatterers(FixedScatterers::s_ptr)                           = 0;

    // Clear all spline scatterers.
    virtual void clear_spline_scatterers()                                              = 0;

    // Add a new set of spline point scatterers.
    virtual void add_spline_scatterers(SplineScatterers::s_ptr)                         = 0;

    // Set scan sequence to use when simulating all RF lines.
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence)               = 0;

    // Set the excitation signal to use when convolving.
    virtual void set_excitation(const ExcitationSignal& new_excitation)                 = 0;

    // Configure an analytical Gaussian beam profile.
    virtual void set_analytical_profile(IBeamProfile::s_ptr beam_profile)               = 0; // TODO: final arguments: float sigma_lateral, float sigma_elevational

    // Configure a lookup table based beam profile.
    virtual void set_lookup_profile(IBeamProfile::s_ptr beam_profile)                   = 0; // TODO: final arguements: ?

    // Simulate all RF lines. Returns vector of IQ samples.
    // Requires that everything is properly configured.
    virtual void simulate_lines(std::vector<std::vector<std::complex<float>> >&  /*out*/ rf_lines) = 0;

    // Get debug data by identifier. Throws std::runtime_error on invalid key.
    virtual std::vector<double> get_debug_data(const std::string& identifier) const = 0;

    // Get the total number of scatterers (fixed and dynamic)
    virtual size_t get_total_num_scatterers() const = 0;

    // Set log object to use (optional)
    virtual void set_logger(ILog::ptr log_object) = 0;
};

// Factory function for creating simulator instances.
// Valid types are:
//     "cpu"   - CPU implementation
//     "gpu"   - GPU implementation
IAlgorithm::s_ptr DLL_PUBLIC Create(const std::string& sim_type);

}   // namespace
