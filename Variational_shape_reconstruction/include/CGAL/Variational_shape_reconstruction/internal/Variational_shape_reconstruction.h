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
#include "clustering.h"
#include "trianglefit.h"

namespace qem {
typedef typename std::unordered_set<std::pair<int, int>, HashPairIndex, EqualPairIndex> IntPairSet; 
typedef std::vector<IntList> IntListList;
typedef CGAL::Bbox_3 Bbox;

typedef typename CGenerator<Kernel> Generator;


class Variational_shape_reconstruction
{
    private:

        // Partition
        std::map<int, int> m_vlabels;  // map of point index to its generator's label
        double m_euclidean_distance_weight = 0.1; // lambda term in region growing

        // generators
        std::vector<Generator> m_generators;  // vector of generators
        int m_nb_generators = 50;  // number of generators 
        //IntList m_generators_count;    
        INIT_QEM_GENERATORS m_init_qem_generators = INIT_QEM_GENERATORS::RANDOM; // method of initialization of generators

        // geometry
        PwiList m_points;  // vector of points with their indices
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
        std::shared_ptr<Clustering> m_pClustering;

        // verbosity level
        VERBOSE_LEVEL m_verbose_level = VERBOSE_LEVEL::HIGH;
        

    public:

    Variational_shape_reconstruction()
    {
    }

    Variational_shape_reconstruction(const Pointset& pointset,
        const int nb_generators,
        const FT euclidean_distance_weight,
        const VERBOSE_LEVEL verbose_level,
        const INIT_QEM_GENERATORS init_qem_generator) :
    m_pointset(pointset),
    m_nb_generators(nb_generators),
    m_init_qem_generators(init_qem_generator),
    m_verbose_level(verbose_level),
    m_euclidean_distance_weight(euclidean_distance_weight)
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
        m_pClustering = std::make_shared<Clustering>(pointset, m_num_knn, m_euclidean_distance_weight, m_verbose_level);

        // init QEM per point and per "vertex" (a point and its neighborhood)
        m_pClustering->initialize_qem_per_point(m_tree);
        m_pClustering->initialize_qem_per_vertex(m_tree);

        // init generators
        initialize_generators();

        // compute connected components and add generators in components without generators
        m_pClustering->compute_connected_and_add_generators(m_generators);

        // init euclidian_dist_weight
        // why are we asking the dist_weight from the user instead of initializing it using the following method??
        m_euclidean_distance_weight = m_num_knn * m_spacing * m_spacing; 

        init_generator_qems(); // init qem of generators and related optimal locations
    }

    void init_generator_qems()
    {
        std::vector<Generator>::iterator it;
        for (it = m_generators.begin(); it != m_generators.end(); it++)
        {
            Generator& generator = *it;
            const int point_index = generator.point_index(); // index of generator point in point set
            QEM_metric& generator_qem = vqem(point_index);
            Point& generator_point = m_pointset.point(point_index);
            generator.location() = compute_optimal_point(generator_qem , generator_point);  // optimal location of generator
            generator.qem() = generator_qem;
        }

         //compute the nearest point from the optimal locations of all generators
        for (int label = 0; label < m_generators.size(); label++) {
            Generator& generator = m_generators[label];
            const Point generator_point = generator.location();
            K_neighbor_search search(m_tree, generator_point, 1);
            for (KNNIterator it = search.begin(); it != search.end(); it++)
            {
                generator.point_index() = (it->first).second;
                generator.location() = m_pointset.point(generator.point_index());
            }
        }
    }

    QEM_metric& vqem(const int index)
    {
        return m_pClustering->vqem(index);
    }

	void initialize_generators()
	{
		std::cout << "Initialization with " << m_nb_generators << " generators" << std::endl;
		switch (m_init_qem_generators)
		{
            /*
		case INIT_QEM_GENERATORS::KMEANS_PLUSPLUS:
			init_generators_kmeanspp(nb_generators);
			if (m_verbose_level != VERBOSE_LEVEL::LOW)
				std::cout << "Initialization method : KMEANS_PLUSPLUS" << std::endl;
			break;
            */
		case INIT_QEM_GENERATORS::FARTHEST:
			init_generators_farthest(m_nb_generators);
			if (m_verbose_level != VERBOSE_LEVEL::LOW)
				std::cout << "Initialization method : FARTHEST" << std::endl;
			break;
		default:
			init_random_generators(m_nb_generators);
			if (m_verbose_level != VERBOSE_LEVEL::LOW)
				std::cout << "Initialization method : RANDOM" << std::endl;
			break;
		}
	}




    /// @brief load the points from a pointset to the m_points list
    /// @param pointset 
    void load_points(const Pointset& pointset)
    {
        for(int point_index = 0; point_index < pointset.size(); point_index++)
        {
            const Point point = pointset.point(point_index);
            m_points.push_back(std::make_pair(point, point_index));
        }
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cout << "Number of points: " << m_pointset.size() << std::endl;
    }

    /// @brief Compute the bounding box of the pointset
    void compute_bounding_box()
    {
        // find bounding box
        // todoquestion : boost bbox over poinset 
        boost::function<Point(Point_with_index &)> pwi_it_to_point_it = boost::bind(&Point_with_index::first, _1);
        m_bbox = CGAL::bbox_3(  boost::make_transform_iterator(m_points.begin(), pwi_it_to_point_it), 
                                boost::make_transform_iterator(m_points.end(), pwi_it_to_point_it));
        m_diag = std::sqrt(CGAL::squared_distance(Point((m_bbox.min)(0), (m_bbox.min)(1), (m_bbox.min)(2)),
            Point((m_bbox.max)(0), (m_bbox.max)(1), (m_bbox.max)(2))));
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cout << "Diagonal of bounding box: " << m_diag << std::endl;
    }
    
    /// @brief Compute the number of generators per connected component of the kNN graph of the pointcloud
    /// return an IntList containing the number of generators corresponding to each component index
    IntList get_generators_per_component()
    {
        IntList generators_per_component = m_pClustering->get_generators_per_component(m_generators);
        for (int component_index = 0; component_index < generators_per_component.size(); component_index++)
            std::cout<< "component " << component_index << " generators count : " << generators_per_component[component_index] << std::endl;
        return generators_per_component;
    }

    /// @brief Compute the number of connected components in the kNN graph of the pointcloud
    /// return an integer: the number of connected components
    int get_component_count()
    {
        return m_pClustering->get_component_count();
    }

    bool is_partionning_valid()
    {
        return m_vlabels.size() == m_pointset.size();
    }

    void set_knn(int num_knn)
    {
        m_num_knn = num_knn;
    }

    void set_distance_weight(int euclidean_distance_weight)
    {
        m_pClustering->set_distance_weight(euclidean_distance_weight);
    }

    /// @brief Initialize with random generators
    void init_random_generators(const std::size_t nb_generators)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        IntList num_range(m_pointset.size());
        std::iota(num_range.begin(), num_range.end(), 0); // fill range with increasing values

        // random number generation and shuffling
        // using the current time as the seed for random seeding 
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 random_generator(seed);
        std::shuffle(num_range.begin(), num_range.end(), random_generator); // shuffling using mersenne twister random number generator

        std::set<int> selected_indices; // set to get them sorted
        for(int point_index = 0; point_index < nb_generators; point_index++)
        {
            selected_indices.insert(num_range[point_index]);
        }
            
        for(int selected_index : selected_indices) 
        {
            m_generators.push_back(Generator(selected_index, m_pointset.point(selected_index)));
        }

        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cout << "Number of random generators: " << m_generators.size() << std::endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cerr << "Random generators in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    }
    
    /*
    /// @brief Initiatlize starting with one random generator and kmeans++
    void init_generators_kmeanspp(const std::size_t nb_generators)
    {
        IntList num_range(m_pointset.size());
        std::iota(num_range.begin(), num_range.end(), 0);
        std::set<int> selected_indices;
        // one generator for k means ++
        for(int i = 0; i < 1; i++)
            selected_indices.insert(num_range[i]);
            
        for(auto &elem: selected_indices) {
            m_generators.push_back(elem);
        }   
        std::mt19937 gen;
        for(int i = 1; i < nb_generators; i++)
        {
            std::vector<float> distance_list;
            for( Pointset::const_iterator m_pointsetit = m_pointset.begin(); m_pointsetit != m_pointset.end(); ++ m_pointsetit )
            {
                const auto point = m_pointset.point(*m_pointsetit);
                float distance = std::numeric_limits<float>::max();
                // we iterate over previously selected generators
                for(int j = 0 ; j < m_generators.size(); j++)
                {
                    double distance_to_generator = CGAL::squared_distance(m_pointset.point(m_generators[j]), point);
                    distance = std::min(distance, distance_to_generator);
                }
                distance_list.push_back(distance);
            }
            // We draw in a random distribution the next generator
            std::discrete_distribution<int> dist(std::begin(distance_list), std::end(distance_list));
            gen.seed(time(0));
            size_t index_max_distance = dist(gen);

            m_generators.push_back(index_max_distance);
        }
    }
    */

    /// @brief Initiatlize starting with one random generator and kmeans farthest
    // FIXME: use farthest point seeding of computational geometry
    void init_generators_farthest(const std::size_t nb_generators)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        IntList num_range(m_pointset.size());
        std::iota(num_range.begin(), num_range.end(), 0); // fill range with increasing values

        // random shuffler
        std::mt19937 random_generator(27);
        std::shuffle(num_range.begin(), num_range.end(), random_generator);

        std::set<int> selected_indices;

        // first generator for k means farthest
        for(int point_index = 0; point_index < 1; point_index++)
            selected_indices.insert(num_range[point_index]);

        for(int selected_index : selected_indices)
            m_generators.push_back(Generator(selected_index, m_pointset.point(selected_index)));

        for (int generator_index = 1; generator_index < nb_generators; generator_index++)
        {
            // fixme: use FT
            std::vector<FT> distance_list;

            for (int point_index = 0; point_index < m_pointset.size(); point_index++)
            {
                Point &point = m_pointset.point(point_index);
                FT distance = FT(1e308);
                // we iterate over previously selected generators, calculate each point's distance from the set of current generators
                for(int prev_generator_index = 0 ; prev_generator_index < m_generators.size(); prev_generator_index++)
                {
                    Point prev_generator_point = m_pointset.point(m_generators[prev_generator_index].point_index());
                    FT distance_to_generator = FT(CGAL::squared_distance(prev_generator_point, point));
                    distance = (std::min)(distance, distance_to_generator);
                }
                distance_list.push_back(distance);
            }

            // We take the farthest point as new generator
            auto farthest_point_iterator = std::max_element(distance_list.begin(), distance_list.end());
            int index_farthest_point = farthest_point_iterator - distance_list.begin();
            m_generators.push_back(Generator(index_farthest_point, m_pointset.point(index_farthest_point)));
        }

        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cout << "Number of generators: " << m_generators.size() << std::endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cerr << "Generators in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
        
    }

    /// @brief partition
    void partition()
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        m_vlabels.clear();
        m_pClustering->partition(m_vlabels, m_generators, true);  

        assert(m_vlabels.size() == m_pointset.size());
    }
    
    /// @brief update generators
    /// return true if generators have changed
    bool update_generators()
    {
        return m_pClustering->update_generators(m_vlabels, m_generators, m_tree);
    }

    /// @brief partition and update generators for n user-defined steps 
    /// @param steps 
    void partition_and_update_generators(size_t steps)
    {
        bool flag = true;
        for(int iteration = 0; iteration < steps && flag; iteration++)
        {
            partition();
            flag = this->update_generators();
        }
    }

    /// @brief computing errors
    /// returns a pair of total summed error and variance of errors
    std::pair <double, double> compute_clustering_errors()
    {
        return m_pClustering->compute_errors(m_vlabels, m_generators);
    }

    void save_clustering_to_ply(std::string& filename)
    {
		std::ofstream file;
        file.open(filename);

        file << "ply\n"
			<< "format ascii 1.0\n"
			<< "element vertex " << m_pointset.size() << "\n"
			<< "property float x\n"
			<< "property float y\n"
			<< "property float z\n"
			<< "property uchar red\n"
			<< "property uchar green\n"
			<< "property uchar blue\n"
			<< "end_header\n";

            std::vector<Vector> colors;
            for (int i = 0; i < m_generators.size(); i++)
            {
                double r = (double)rand() / (RAND_MAX);
                double g = (double)rand() / (RAND_MAX);
                double b = (double)rand() / (RAND_MAX);
                colors.push_back(Vector(r, g, b));
            }

            for (int point_index = 0; point_index < m_pointset.size(); point_index++)
            {
                if (m_vlabels.find(point_index) == m_vlabels.end())
                    continue;

                Vector& color = colors[m_vlabels[point_index]];

                Point point = m_pointset.point(point_index);
                file << point.x() << " " << point.y() << " " << point.z() << " ";
                int r = static_cast<int>(255 * color.x());
                int g = static_cast<int>(255 * color.y());
                int b = static_cast<int>(255 * color.z());
                file << r << " " << g << " " << b << "\n";
            }

            file.close();
    }


    /// @brief Split the cluster with the split_ratio
    /// @param split_ratio 
    /// @param iteration 
    /// @return total of added generators
    size_t guided_split_clusters(const double split_ratio, const int iteration) // batch splitting
    {
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cout << "Begin guided split..." << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto nb_generators = m_pClustering->guided_split_clusters(m_vlabels,m_generators,
            m_diag,m_spacing,split_ratio, iteration);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        if(m_verbose_level != VERBOSE_LEVEL::LOW)
            std::cerr << "Guided split in " << elapsed << "[us]" << std::endl;

        return nb_generators;
    }

    /// @brief automatic clustering
    /// @param steps maximum number of iterations
    /// @param split_ratio weight of split_threshold between 0 and 1
    void clustering(const size_t steps, const double split_ratio)
    {
        int nb_generators = 0;
        int iteration = 0;

        while(iteration < steps)
        {
            partition_and_update_generators(steps);
            nb_generators = m_nb_generators;
            m_nb_generators = guided_split_clusters(split_ratio, iteration++);
        }
    }

    /// @brief automatic clustering and guided split clusters
    /// @param steps maximum number of iterations
    /// @param split_ratio weight of split_threshold between 0 and 1
    void clustering_and_compute_errors(const size_t steps, const double split_ratio, std::ofstream& file, const bool save_clustering_to_ply)
    {
        int nb_generators = 0;
        int iteration = 0;

        while (iteration < steps && m_nb_generators!=nb_generators)
        {
            std::cout << "m_generators.size(): " << m_generators.size() << std::endl;
            std::cout << "m_nb_generators: " << m_nb_generators << std::endl;

            // FIXME
            partition_and_update_generators(steps);
            nb_generators = m_nb_generators;
            m_nb_generators = nb_generators + guided_split_clusters(split_ratio, iteration++);
            const std::pair<double, double> total_error = compute_clustering_errors();
            file << iteration << "," << total_error.first << "," << total_error.second << std::endl;

            // save clustering to file every 10 iterations
            if (save_clustering_to_ply && iteration % 10 == 0)
            {
                std::string filename("clustering-");
                filename.append(std::to_string(iteration));
                filename.append(std::string(".ply"));
                this->save_clustering_to_ply(filename); 
            }
        }
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
        if(lu_decomp.isInvertible())
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

    // added this briefly for running test files
    // fixme: if this is only for debugging, then comment and prefix
    Pointset get_point_cloud_clustered()
    {
        Pointset pointset;
        std::vector<Vector> colors;
        for (int k = 0; k < m_generators.size();k++)
        {
            const double r = (double)rand() / RAND_MAX;
            const double g = (double)rand() / RAND_MAX;
            const double b = (double)rand() / RAND_MAX;
            colors.push_back(Vector(r, g, b));
        }
        for (int i = 0; i < m_points.size();i++)
            pointset.insert(m_points[i].first, colors[m_vlabels[i]]);

        return pointset;
    }

    const Polyhedron& get_reconstructed_mesh()
    {
        return m_triangle_fit.get_mesh();
    }
    void write_csv()
    {
        m_pClustering->write_csv();
    }

    // reconstruction 
    bool reconstruction(const double dist_ratio, 
        const double fitting, 
        const double coverage, 
        const double complexity, 
        const bool use_soft_reconstruction = false)
    {
        if(!create_adjacent_edges()) 
            return false;

        if(!create_candidate_facets()) 
            return false;

        mlp_reconstruction(dist_ratio, fitting, coverage, complexity);
        
        auto valid = m_triangle_fit.get_mesh().is_valid();
        if(!valid && use_soft_reconstruction)
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
        if(m_generators.empty())
        {
            if(m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "No generators" << std::endl;
            return false;
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::vector<Point> generator_locations; 
        for (int i = 0; i < m_generators.size(); i++)
            generator_locations.push_back(m_generators[i].location());

        // search for adjacent clusters
        IntPairSet adjacent_pairs;
        for(int i = 0; i < m_pointset.size(); i++)
        {
            // skip unlabeled points
            if(m_vlabels.find(i) == m_vlabels.end())
                continue;

            int point_label = m_vlabels[i];
            K_neighbor_search search(m_tree, m_pointset.point(i), m_num_knn);

            for(KNNIterator it = search.begin(); it != search.end(); it++)
            {
                int neighor_index = (it->first).second;

                if(m_vlabels.find(neighor_index) != m_vlabels.end())
                {
                    int neighbor_label = m_vlabels[neighor_index];
                    if(point_label != neighbor_label)
                        adjacent_pairs.insert(std::make_pair(point_label, neighbor_label));
                }              
            }
        }
        const bool valid = !adjacent_pairs.empty();

        if(valid)
            m_triangle_fit.initialize_adjacent_graph(generator_locations, adjacent_pairs, m_bbox, m_diag);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
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
        if(m_verbose_level != VERBOSE_LEVEL::LOW)
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

        std::transform( m_points.begin(), 
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

        std::transform( m_points.begin(), 
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