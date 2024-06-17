// Copyright (c) 2023 GeometryFactory
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s) : Tong Zhao, Mathieu Ladeuil, Pierre Alliez

#include "io.h"
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Splitters.h>

#include "generator.h"
#include "clustering_ftc.h"
#include "trianglefit.h"

namespace qem {
    typedef typename std::unordered_set<std::pair<int, int>, HashPairIndex, EqualPairIndex> IntPairSet;
    typedef std::vector<IntList> IntListList;
    typedef CGAL::Bbox_3 Bbox;

    typedef typename CGenerator<Kernel> Generator;

    /*
    ***********************************************************************************************************************************************
    ***********************************************************************************************************************************************
    ***********************************************************************************************************************************************
    ***********************************************************************************************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************         FINE          *********************************************************************
    ***************************************************          TO           *********************************************************************
    ***************************************************        COARSE         *********************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************                       *********************************************************************
    ***************************************************                       *********************************************************************
    ***********************************************************************************************************************************************
    ***********************************************************************************************************************************************
    ***********************************************************************************************************************************************
    ***********************************************************************************************************************************************
    */

    class Variational_shape_reconstruction_ftc
    {
    private:

        // Clustering
        std::map<int, int> m_vlabels;  // map of point index to its generator's label
        std::vector< std::vector<int> > m_pindices; // vector of point indices in each generator's cluster
        int m_nb_clusters = 1e3;

        // generators
        std::vector<Generator> m_generators;  // vector of generators 

        // points
        PwiList m_points;      // list of points and their indices
        Pointset m_pointset;   // set of points and their normals

        // KNN tree
        KNNTree m_tree;  // the tree of all points
        unsigned int m_num_knn = 12;  // the value of k in k nearest neighbours

        // init
        Bbox m_bbox;
        double m_diag;
        double m_spacing;

        // mesh reconstruction
        TriangleFit m_triangle_fit;

        // clustering
        std::shared_ptr<Clustering_ftc> m_pClustering_ftc;

        // verbosity level
        VERBOSE_LEVEL m_verbose_level = VERBOSE_LEVEL::HIGH;


    public:

        Variational_shape_reconstruction_ftc()
        {
        }

        Variational_shape_reconstruction_ftc(const Pointset& pointset,
            const int nb_clusters,
            const int num_knn,
            const VERBOSE_LEVEL verbose_level) :
            m_pointset(pointset),
            m_num_knn(num_knn),
            m_nb_clusters(nb_clusters),
            m_verbose_level(verbose_level)
        {
            load_points(m_pointset);
            compute_bounding_box();

            // init KD tree
            m_tree.clear();
            m_tree.insert(m_points.begin(), m_points.end());

            // compute average spacing
            m_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(m_points, m_num_knn,
                CGAL::parameters::point_map(CGAL::First_of_pair_property_map<std::pair<Point, std::size_t> >()));

            // clustering instantiation
            m_pClustering_ftc = std::make_shared<Clustering_ftc>(m_pointset, m_num_knn, m_nb_clusters, m_verbose_level);

            // init QEM per point and per "vertex" (a point and its neighborhood)
            m_pClustering_ftc->initialize_qem_per_point(m_tree);
            m_pClustering_ftc->initialize_qem_per_vertex(m_tree);

            // init generators
            initialize_generators();
            init_generator_qems(); // init qem of generators and related optimal locations

            // init priority queue
            m_pClustering_ftc->init_pqueue(m_vlabels,m_pindices,m_generators);
        }

        /// @brief load the points from a pointset to the m_points list
        /// @param pointset 
        void load_points(const Pointset& pointset)
        {
            for (int point_index = 0; point_index < pointset.size(); point_index++)
            {
                const Point point = pointset.point(point_index);
                m_points.push_back(std::make_pair(point, point_index));
            }
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Number of points: " << m_pointset.size() << std::endl;
        }

        /// @brief Compute the bounding box of the pointset
        void compute_bounding_box()
        {
            // find bounding box
            // todoquestion : boost bbox over poinset 
            boost::function<Point(Point_with_index&)> pwi_it_to_point_it = boost::bind(&Point_with_index::first, _1);
            m_bbox = CGAL::bbox_3(boost::make_transform_iterator(m_points.begin(), pwi_it_to_point_it),
                boost::make_transform_iterator(m_points.end(), pwi_it_to_point_it));
            m_diag = std::sqrt(CGAL::squared_distance(Point((m_bbox.min)(0), (m_bbox.min)(1), (m_bbox.min)(2)),
                Point((m_bbox.max)(0), (m_bbox.max)(1), (m_bbox.max)(2))));
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Diagonal of bounding box: " << m_diag << std::endl;
        }

        void initialize_generators()
        {
            for (int point_index = 0; point_index < m_pointset.size(); point_index++) {
                m_pindices.push_back(std::vector<int>());
                m_generators.push_back(Generator(point_index, m_pointset.point(point_index)));
                int label_generator = m_generators.size() - 1;
                m_vlabels[point_index] = label_generator;
                m_pindices[label_generator].push_back(point_index);
            }
        }

        void init_generator_qems()
        {
            std::vector<Generator>::iterator it;
            for (it = m_generators.begin(); it != m_generators.end(); it++)
            {
                Generator& generator = *it;
                const int point_index = generator.point_index(); // index of generator point in point set
                QEM_metric& generator_qem = vqem(point_index);
                generator.qem() = generator_qem;
                Point& generator_point = m_pointset.point(point_index);
                generator.location() = compute_optimal_point(generator_qem, generator_point);  // optimal location of generator
            }
        }

        QEM_metric& vqem(const int index)
        {
            return m_pClustering_ftc->vqem(index);
        }

        bool is_partionning_valid()
        {
            return m_vlabels.size() == m_pointset.size();
        }

        void set_knn(int num_knn)
        {
            m_num_knn = num_knn;
        }

        /// @brief computing errors
        /// returns a pair of total summed error and variance of errors
        std::pair <double, double> compute_clustering_errors()
        {
            return m_pClustering_ftc->compute_errors(m_generators);
        }


        /// @brief automatic clustering
        void clustering()
        {
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Begin clustering..." << std::endl;

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            m_pClustering_ftc->clustering(m_vlabels, m_pindices, m_generators);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cerr << "Clustering in " << elapsed << "[us]" << std::endl;
        }

        /// @brief automatic clustering, computing errors and storing clusterings to .ply files
        /// @param file file stream for storing errors
        void clustering_and_compute_errors(std::ofstream& file)
        {
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Begin clustering and compute errors..." << std::endl;

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            m_pClustering_ftc->clustering_and_compute_errors(file, m_vlabels, m_pindices, m_generators);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cerr << "Clustering and compute errors in " << elapsed << "[us]" << std::endl;
        }

        Point compute_optimal_point(QEM_metric& cluster_qem, Point& initial_guess)
        {
            // solve Qx = b
            Eigen::MatrixXd qem_mat = cluster_qem.get_4x4_svd_matrix();
            Eigen::VectorXd qem_vec = qem_mat.row(3); // 0., 0., 0., 1.
            Eigen::VectorXd optim(4);

            // check rank
            Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(qem_mat);
            lu_decomp.setThreshold(1e-5);

            // full rank -> direct inverse
            if (lu_decomp.isInvertible())
            {
                optim = lu_decomp.inverse() * qem_vec;
            }
            else
            {   // low rank -> svd pseudo-inverse
                Eigen::JacobiSVD<Eigen::MatrixXd> svd_decomp(qem_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
                svd_decomp.setThreshold(1e-5);

                optim(0) = initial_guess.x();
                optim(1) = initial_guess.y();
                optim(2) = initial_guess.z();
                optim(3) = 1.;

                optim = optim + svd_decomp.solve(qem_vec - qem_mat * optim);
            }

            Point optim_point(optim(0), optim(1), optim(2));

            return optim_point;
        }

        const Polyhedron& get_reconstructed_mesh()
        {
            return m_triangle_fit.get_mesh();
        }

        // reconstruction 
        bool reconstruction(const double dist_ratio,
            const double fitting,
            const double coverage,
            const double complexity,
            const bool use_soft_reconstruction = false)
        {
            if (!create_adjacent_edges())
                return false;

            if (!create_candidate_facets())
                return false;

            mlp_reconstruction(dist_ratio, fitting, coverage, complexity);

            auto valid = m_triangle_fit.get_mesh().is_valid();
            if (!valid && use_soft_reconstruction)
            {
                std::cout << "Manifold reconstruction failed, run non-manifold variant" << std::endl;
                non_manifold_reconstruction(dist_ratio, fitting, coverage, complexity);
                valid = m_triangle_fit.get_mesh().is_valid();
            }
            return valid;
        }

        bool create_adjacent_edges()
        {
            // check generators
            if (m_generators.empty())
            {
                if (m_verbose_level != VERBOSE_LEVEL::LOW)
                    std::cout << "No generators" << std::endl;
                return false;
            }

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            std::vector<Point> generator_locations;
            for (int i = 0; i < m_generators.size(); i++)
                generator_locations.push_back(m_generators[i].location());

            // search for adjacent clusters
            IntPairSet adjacent_pairs;
            for (int i = 0; i < m_pointset.size(); i++)
            {
                // skip unlabeled points
                if (m_vlabels.find(i) == m_vlabels.end())
                    continue;

                int label_generator = m_vlabels[i];
                K_neighbor_search search(m_tree, m_pointset.point(i), m_num_knn);

                for (KNNIterator it = search.begin(); it != search.end(); it++)
                {
                    int neighor_index = (it->first).second;

                    if (m_vlabels.find(neighor_index) != m_vlabels.end())
                    {
                        int label_neighbor_generator = m_vlabels[neighor_index];
                        if (label_generator != label_neighbor_generator)
                            adjacent_pairs.insert(std::make_pair(label_generator, label_neighbor_generator));
                    }
                }
            }
            const bool valid = !adjacent_pairs.empty();

            if (valid)
                m_triangle_fit.initialize_adjacent_graph(generator_locations, adjacent_pairs, m_bbox, m_diag);

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cerr << "Candidate edges in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

            return valid;
        }

        void update_adjacent_edges(std::vector<float>& adjacent_edges)
        {
            m_triangle_fit.update_adjacent_edges(adjacent_edges);
        }

        bool create_candidate_facets()
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            bool valid = m_triangle_fit.create_candidate_facets();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cerr << "Candidate facet in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
            return valid;
        }

        void update_candidate_facets(std::vector<float>& candidate_facets, std::vector<float>& candidate_normals)
        {
            m_triangle_fit.update_candidate_facets(candidate_facets, candidate_normals);
        }

        void mlp_reconstruction(double dist_ratio, double fitting, double coverage, double complexity)
        {
            std::vector<Point> input_point_set;

            std::transform(m_points.begin(),
                m_points.end(),
                std::back_inserter(input_point_set),
                [](const Point_with_index& p) { return p.first; });

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            m_triangle_fit.reconstruct(input_point_set, m_spacing, dist_ratio, fitting, coverage, complexity);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cerr << "MIP solver in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

        }
        void non_manifold_reconstruction(double dist_ratio, double fitting, double coverage, double complexity)
        {
            std::vector<Point> input_point_set;

            std::transform(m_points.begin(),
                m_points.end(),
                std::back_inserter(input_point_set),
                [](const Point_with_index& p) { return p.first; });

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            m_triangle_fit.nonmanifold_reconstruct(input_point_set, m_spacing, dist_ratio, fitting, coverage, complexity);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cerr << "Non manifold solver in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

        }
        void update_fit_surface(std::vector<float>& fit_facets, std::vector<float>& fit_normals)
        {
            m_triangle_fit.update_fit_surface(fit_facets, fit_normals);
        }

        void update_fit_soup(std::vector<float>& fit_soup_facets, std::vector<float>& fit_soup_normals)
        {
            m_triangle_fit.update_fit_soup(fit_soup_facets, fit_soup_normals);
        }
    };
}