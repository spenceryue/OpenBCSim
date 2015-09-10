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

#include <random>
#include "ScattererModel.hpp"

void BaseScattererModel::precomputeCube() {
    QVector<GLfloat> ps; // points
    QVector<GLfloat> ns; // normals

    // triangle 1
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);      ns.push_back(-1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);       ns.push_back(-1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(1.0f);        ns.push_back(-1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    // triangle 2
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);        ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(-1.0f);
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);      ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(-1.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);       ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(-1.0f);
    // triangle 3
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(-1.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);      ns.push_back(0.0f); ns.push_back(-1.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);       ns.push_back(0.0f); ns.push_back(-1.0f); ns.push_back(0.0f);
    // triangle 4
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);        ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(-1.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);       ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(-1.0f);
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);      ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(-1.0f);
    // triangle 5
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);      ns.push_back(-1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(1.0f);        ns.push_back(-1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);       ns.push_back(-1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    // triangle 6
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(-1.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);       ns.push_back(0.0f); ns.push_back(-1.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);       ns.push_back(0.0f); ns.push_back(-1.0f); ns.push_back(0.0f);
    // triangle 7
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(1.0f);
    ps.push_back(-1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);       ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(1.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(1.0f);
    // triangle 8
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(1.0f);         ns.push_back(1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);       ns.push_back(1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);        ns.push_back(1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    // triangle 9
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(-1.0f);       ns.push_back(1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(1.0f);         ns.push_back(1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);        ns.push_back(1.0f); ns.push_back(0.0f); ns.push_back(0.0f);
    // triangle 10
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(1.0f);         ns.push_back(0.0f); ns.push_back(1.0f); ns.push_back(0.0f);
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);        ns.push_back(0.0f); ns.push_back(1.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);       ns.push_back(0.0f); ns.push_back(1.0f); ns.push_back(0.0f);
    // triangle 11
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(1.0f);         ns.push_back(0.0f); ns.push_back(1.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(-1.0f);       ns.push_back(0.0f); ns.push_back(1.0f); ns.push_back(0.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(1.0f); ns.push_back(0.0f);
    // triangle 12
    ps.push_back(1.0f); ps.push_back(1.0f); ps.push_back(1.0f);         ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(1.0f);
    ps.push_back(-1.0f); ps.push_back(1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(1.0f);
    ps.push_back(1.0f); ps.push_back(-1.0f); ps.push_back(1.0f);        ns.push_back(0.0f); ns.push_back(0.0f); ns.push_back(1.0f);
        
    m_cube_points = ps;
    m_cube_normals = ns;
}


void SplineScattererModel::setTimestamp(float timestamp) {
    size_t num_splines = m_splines.size();
    m_data.clear();
        
    // Each scatterer has one point and one normal vector
    m_data.reserve(num_splines*2*3);
    
    // Evaluate all splines in timestamp
    const float radius = 0.1e-3;
    for (size_t i = 0; i < num_splines; i++) {
        // Evaluate scatterer position
        bcsim::vector3 p = RenderCurve<float, bcsim::vector3>(m_splines[i], timestamp);
        
        // Add a correctly positioned cube in each scatterer position
        for (size_t i = 0; i < m_cube_points.size()/3; i++) {
            m_data.push_back(m_cube_points[3*i]  *radius + p.x);
            m_data.push_back(m_cube_points[3*i+1]*radius + p.y);
            m_data.push_back(m_cube_points[3*i+2]*radius + p.z);
            m_data.push_back(m_cube_normals[3*i]);
            m_data.push_back(m_cube_normals[3*i+1]);
            m_data.push_back(m_cube_normals[3*i+2]);
        }
    }
}

void SplineScattererModel::setSplines(const std::vector<SplineCurve<float, bcsim::vector3> >& splines) {
    size_t num_splines = splines.size();
    m_splines = splines;
    m_scatterer_normals = generateRandomNormalVectors(num_splines);
}

QVector<QVector3D> BaseScattererModel::generateRandomNormalVectors(int num_vectors) {
    QVector<QVector3D> res(num_vectors);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.2, 0.2);
    for (int i = 0; i < num_vectors; i++) {
        auto v = QVector3D(dis(gen), dis(gen), dis(gen));
        v.normalize();
        res[i] = v;
    }
    return res;
}

void FixedScattererModel::setPoints(const std::vector<bcsim::vector3>& points) {
    size_t num_points = points.size();
    m_data.clear();
        
    // Each scatterer has one point and one normal vector
    m_data.reserve(num_points*2*3);
    
    // Evaluate all splines in timestamp
    const float radius = 0.1e-3;
    for (size_t i = 0; i < num_points; i++) {
        auto p = points[i];
        
        // Add a correctly positioned cube in each scatterer position
        for (size_t i = 0; i < m_cube_points.size()/3; i++) {
            m_data.push_back(m_cube_points[3*i]  *radius + p.x);
            m_data.push_back(m_cube_points[3*i+1]*radius + p.y);
            m_data.push_back(m_cube_points[3*i+2]*radius + p.z);
            m_data.push_back(m_cube_normals[3*i]);
            m_data.push_back(m_cube_normals[3*i+1]);
            m_data.push_back(m_cube_normals[3*i+2]);
        }
    }
}
