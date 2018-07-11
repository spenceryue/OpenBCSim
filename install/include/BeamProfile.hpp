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
#include <cmath>
#include <memory>
#include "export_macros.hpp"
#include "BCSimConfig.hpp"

namespace bcsim {

class DLL_PUBLIC IBeamProfile {
public:
    typedef std::shared_ptr<IBeamProfile> s_ptr;
    typedef std::unique_ptr<IBeamProfile> u_ptr;
    
    virtual ~IBeamProfile() { }
    
    /**
      * Return the sensitivity of a beam profile at a position in the
      * local beam coordinate system.
      * \param r    The radial component
      * \param l    The lateral component
      * \param e    The elevational component
      */
    virtual float sampleProfile(float r, float l, float e) = 0;
};

// Analytical infinitely long Gaussian cylinder with an elliptical cross section
// parameterized by a lateral and elevational sigma value.
class DLL_PUBLIC GaussianBeamProfile : public IBeamProfile {
public:
    GaussianBeamProfile(float sigmaLateral, float sigmaElevational);
    
    //Set a new lateral sigma value.
    void setSigmaLateral(float newSigmaLateral);
    
    // Set a new elevational sigma value.
    void setSigmaElevational(float newSigmaElevational);
        
    virtual float sampleProfile(float r, float l, float e);
    
    // HACK needed for current GPU algorithms which needs sigma values as
    // kernel parameters
    float getSigmaLateral()     const { return m_sigma_lateral; }
    float getSigmaElevational() const { return m_sigma_elevational; }

protected:
    void updateCaching();
    
protected:
    float m_sigma_lateral;             // PSF sigma in lateral direction [m] 
    float m_sigma_elevational;         // PSF sigma in elevational direction [m]
    
    // Cached values
    float m_two_sigma_lateral_squared;
    float m_two_sigma_elevational_squared;
};


// BeamProfile with LUT and nearest neighbour interpolation.
// Returns zero value if outside data region.
class DLL_PUBLIC LUTBeamProfile : public IBeamProfile {
public:
    // Define number of samples and geometrical extent of each direction.
    LUTBeamProfile(int num_samples_rad, int num_samples_lat, int num_samples_ele,
                   Interval range_range, Interval lateral_range, Interval elevational_range);
    
    virtual float sampleProfile(float r, float l, float e);
        
    // Set sample based on discrete indices.
    void setDiscreteSample(int ir, int il, int ie, float new_sample);

    Interval getRangeRange() const {
        return m_range_range;
    }

    Interval getLateralRange() const {
        return m_lateral_range;
    }

    Interval getElevationalRange() const {
        return m_elevational_range;
    }

    int getNumSamplesRadial() const {
        return m_num_samples_rad;
    }

    int getNumSamplesLateral() const {
        return m_num_samples_lat;
    }

    int getNumSamplesElevational() const {
        return m_num_samples_ele;
    }

protected:
    // row-major indexing
    // dim0: radial, dim1: lateral, dim2: elevational
    long getIndex(int r, int l, int e) {
        return e+m_num_samples_ele*l+m_num_samples_lat*m_num_samples_ele*r;
    }

protected:
    int m_num_samples_rad;
    int m_num_samples_lat;
    int m_num_samples_ele;
    double m_dr;
    double m_dl;
    double m_de;
    std::vector<float> m_samples;
    Interval m_range_range;
    Interval m_lateral_range;
    Interval m_elevational_range;
};

}   // namespace

