#include "types.h"

// qem
#include "qem.h"
#include "candidate.h"
#include "pqueue.h"
#include "generator.h"

#include <CGAL/bounding_box.h>
#include <CGAL/compute_average_spacing.h>
// knn
#include <CGAL/Orthogonal_k_neighbor_search.h>
 //#include <CGAL/jet_estimate_normals.h>

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/iterator/transform_iterator.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <random>

#include "helper_metrics.h"
#include "io.h"
// #include "color.hpp"

typedef CGAL::Exact_predicates_inexact_constructions_kernel     Kernel;
typedef Kernel::FT                  FT;

// KNN tree
typedef std::pair<Point, std::size_t>                                               Point_with_index;
typedef std::vector<Point_with_index>                                               PwiList;
typedef CGAL::First_of_pair_property_map<Point_with_index>                          Point_map_pwi;
typedef CGAL::Search_traits_3<Kernel>                                               Traits_base;
typedef CGAL::Search_traits_adapter<Point_with_index, Point_map_pwi, Traits_base>   Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>                                  K_neighbor_search;
typedef typename K_neighbor_search::Tree                                            KNNTree;
typedef typename K_neighbor_search::Distance                                        KNNDistance;
typedef typename K_neighbor_search::iterator                                        KNNIterator;

// Priority queue
typedef qem::Candidate<int> CCandidate;
typedef qem::Candidate_more<CCandidate> More;
typedef qem::Custom_priority_queue<CCandidate, More> PQueue;

typedef typename qem::CGenerator<Kernel> Generator;

namespace qem
{   
    class Clustering
    {   

    // //the private variables are initialized below, pasted the section here just for lookup 
    //private:
    //    Pointset m_point_set;  // Pointset of the point cloud
    //    int m_num_knn = 12;    // number of nearest neighbours in k nearest neighbours
    //    double m_dist_weight = 0.1;   // lambda term in region growing

    //    // qem
    //    std::vector<QEM_metric> m_pqems; // qem per point
    //    std::vector<QEM_metric> m_vqems; // diffused qem per point
    //    std::vector<std::vector<int> > m_graph; // neighborhood graph (indices)
    //    std::vector<bool> m_visited; // whether each point is visited or not
    //    std::vector<int> m_component; // component index corresponding to each point index
    //    int component_count = 0; // fixme  // number of components

    //    VERBOSE_LEVEL m_verbose_level = qem::VERBOSE_LEVEL::HIGH;

    //    // csv
    //    std::shared_ptr<DataWriter> csv_writer;


        public:

        Clustering()
        {
        }

        Clustering(const Pointset& pointset,
        unsigned int num_knn,
        double euclidean_distance_weight,
        VERBOSE_LEVEL verbose_level)
        {
            m_point_set = pointset;
            m_num_knn = num_knn;
            m_dist_weight = euclidean_distance_weight;
            m_verbose_level = verbose_level;
            csv_writer = std::make_shared<DataWriter>(pointset.size());
        }

        /// @brief Compute the qem for each point based on the k nearest neighbor neighbors
        // fixme: rather based on the normal and average distance to neighbors!
        // TODO: add function to estimate normals
        void initialize_qem_per_point(const KNNTree& tree)
        {
            // init vector of qems
            m_pqems.clear();

            for(int point_index = 0; point_index < m_point_set.size(); point_index++)
            {
                Point point = m_point_set.point(point_index);
                K_neighbor_search search(tree, point, m_num_knn);  
                KNNDistance tr_dist;  // orthogonal distance : transformed distace in k nearest neighbours graph

                double avg_dist = 0.;
                for(KNNIterator it = search.begin(); it != search.end(); it++)
                    avg_dist += tr_dist.inverse_of_transformed_distance(it->second);
                avg_dist = avg_dist / (double)m_num_knn;

                //if (!m_point_set.has_normal_map()) {
                //    m_point_set.add_normal_map();
                //    CGAL::jet_estimate_normals<CGAL::Sequential_tag>
                //        (m_point_set,
                //        m_num_knn, 
                //        m_point_set.parameters(). // Named parameters provided by Point_set_3
                //        degree_fitting(2));     // additional named parameter specific to jet_estimate_normals
                //}
                
                QEM_metric pqem = compute_qem_for_point(point, m_point_set.normal(point_index), avg_dist * avg_dist / 2);
                m_pqems.push_back(pqem);
            }
        }

        /// @brief Compute the qem for a point weighted by the area of the face
        /// @param query the point to compute the qem for
        /// @param normal of the face
        /// @param area of the face
        /// @return the qem computed
        QEM_metric compute_qem_for_point(const Point& query, const Vector& normal, const double &area)
        {
            QEM_metric qem;
            qem.init_qem_metrics_face(area, query, normal);
            return qem;
        }

        /// @brief Compute the sum of the qem neighbor points for each point in m_vqems
        /// Also build the neighbourhood graph
        /// @param m_tree the knn tree
        void initialize_qem_per_vertex(const KNNTree& m_tree)
        {
            m_vqems.clear();
            m_graph.clear();
            for(int point_index = 0; point_index < m_point_set.size(); point_index++)
            {
                K_neighbor_search search(m_tree, m_point_set.point(point_index), m_num_knn);

                IntList neighbors;

                // init with qem of point itself 
                QEM_metric vqem = m_pqems[point_index];

                for(KNNIterator it = search.begin(); it != search.end(); it++)
                {
                    auto neighbor_idx = (it->first).second; // it->first is of the type Point_with_index
                    vqem = vqem + m_pqems[neighbor_idx];
                    neighbors.push_back(neighbor_idx);
                }
                m_graph.push_back(neighbors);

                m_vqems.push_back(vqem);
            }

        }

        /// @brief Compute the connected components
        /// FIXME: cannot load .ply file
        void compute_connected()
        {
            size_t point_count = m_graph.size(); 
            m_visited.resize(point_count);
            m_component.resize(point_count);
            std::fill(m_visited.begin(), m_visited.end(), false);
            std::fill(m_component.begin(), m_component.end(), 0);
            component_count = 0;

            for(int point_index = 0 ; point_index < point_count; point_index++)
            {
                if(!m_visited[point_index])
                {
                    visit(point_index, component_count);
                    component_count++;
                }
            }
            
            if(m_verbose_level == VERBOSE_LEVEL::HIGH)
            {
                std::ofstream clustering_connected;
                clustering_connected.open("clustering_connected.ply");

                clustering_connected << "ply\n"
                            << "format ascii 1.0\n"
                            << "element vertex " << m_point_set.size() << "\n"
                            << "property float x\n"
                            << "property float y\n"
                            << "property float z\n"
                            << "property uchar red\n"
                            << "property uchar green\n"
                            << "property uchar blue\n"
                            << "end_header\n";

                std::vector<Vector> colors;
                for(int component_index = 0 ; component_index < component_count; component_index++)
                {
                    double r = (double) rand() / (RAND_MAX);
                    double g = (double) rand() / (RAND_MAX);
                    double b = (double) rand() / (RAND_MAX);
                    colors.push_back(Vector(r, g, b));
                }

                for(int point_index = 0; point_index < m_point_set.size(); point_index++)
                {

                    Point point = m_point_set.point(point_index);
                    clustering_connected << point.x() << " " << point.y() << " " << point.z() << " ";
                    Vector color = colors[m_component[point_index]];
                    clustering_connected << static_cast<int>(255.0 * color.x()) << " " << 
                                            static_cast<int>(255.0 * color.y()) << " " << 
                                            static_cast<int>(255.0 * color.z()) << "\n";

                    point_index++;
                }
                clustering_connected.close();
            }

        }

        /// @brief visit function that explore the graph of neighbors using a 
        /// queue to assign a component index to each point (Breadth First Traversal)
        /// @param start_point_index index of the starting point
        /// @param component_index index of the current connected component
        void visit(const int start_point_index, const int component_index)
        {
            std::queue<int> queue;
            queue.push(start_point_index);
            
            while(!queue.empty())
            {
                int current_point_index = queue.front(); queue.pop();
                if (m_visited[current_point_index])
                    continue;

                m_component[current_point_index] = component_index;
                m_visited[current_point_index] = true;
                for(int neighbor_index : m_graph[current_point_index])
                {
                    if(!m_visited[neighbor_index])
                    {
                        queue.push(neighbor_index);
                    }
                }
            }
        }


        // @brief compute connected components and add generators accordingly
        // @param vector of generators
        void compute_connected_and_add_generators(std::vector<Generator>& generators) {
            compute_connected(); // compute connected components
            add_generators(generators); // add generators in components having no generators
        }

        // @brief add generator in component having no generator
        // @param vector of generators
        void add_generators(std::vector<Generator>& generators) {

            if(m_verbose_level != VERBOSE_LEVEL::LOW)
            {
                std::cout << "Number of connected components: " << component_count << std::endl;
                std::cout << "Number of generators: " << generators.size() << std::endl;
            }
            for(int component_index = 0 ; component_index < component_count; component_index++)
            {
                bool found = false;  // if there is a generator in this component
                for(int label_generator = 0 ; label_generator < generators.size(); label_generator++)
                {
                    int generator_index = generators[label_generator].point_index();
                    if(component_index == m_component[generator_index])
                    {
                        found = true;
                        break;
                    }
                }

                if(!found)
                {
                    // Add a new generator in the connected component without generator
                    // Make point in the component having the smallest point index the generator
                    // FIXME: make the point in the component farthest away from the set of generators the new generator 
                    // not really required because chance is quite for no generator in a component
                    int generator_index = std::find(m_component.begin(), m_component.end(), component_index) - m_component.begin();
                    generators.push_back(Generator(generator_index, m_point_set.point(generator_index)));
                    if(m_verbose_level != VERBOSE_LEVEL::LOW)
                        std::cout << "Added generator in component number: " << component_index << std::endl;
                }
            }

            if(m_verbose_level == VERBOSE_LEVEL::HIGH)
            {
                // Create a pointcloud of the graph of neighbors so that
                // each point is connected to each of his neighbors
                std::ofstream neighbors_graph;
                neighbors_graph.open("neighbors_graph.ply");

                std::size_t sum = 0;
                for (auto &&i : m_graph) {
                    sum += i.size();
                }

                neighbors_graph << "ply\n"
                        << "format ascii 1.0\n"
                        << "element vertex " << m_point_set.size() << "\n"
                        << "property float x\n"
                        << "property float y\n"
                        << "property float z\n"
                        << "element face " << sum << "\n"
                        << "property list uchar int vertex_indices\n"
                        << "end_header\n";

                for(Pointset::const_iterator it = m_point_set.begin(); it != m_point_set.end(); ++ it)
                {
                    auto point = m_point_set.point(*it);
                    neighbors_graph << point.x() << " " << point.y() << " " << point.z() << std::endl;
                }

                for(int i = 0; i < m_graph.size(); i++)
                {
                    for(int j = 0; j < m_graph[i].size(); j++)
                    {
                        neighbors_graph << "2 "<<i<<" "<<m_graph[i][j]<<"\n";  // what is 2 here?
                    }
                }
                neighbors_graph.close();
            }
        }

        /// @brief number of generators per component
        /// @param vector of generators
        /// return a vector containing the number of generators corresponding to each component index
        IntList get_generators_per_component(std::vector<Generator>& generators)  
        {
            IntList generators_per_component(component_count,0);
            for(int label_generator = 0 ; label_generator < generators.size(); label_generator++)
            {
                int generator_point_index = generators[label_generator].point_index();
                int component_index = m_component[generator_point_index];
                generators_per_component[component_index]++; 
            }

            return generators_per_component;
        }

        /// @brief Partition: find the best generator for each point, and update cluster QEM
        /// via region growing and the cost function compute_growing_error
        /// @param m_vlabels 
        /// @param generators_qem 
        /// @param generators 
        /// @param flag_dist 
        void partition(std::map<int, int>& m_vlabels,
            std::vector<Generator>& generators,
            const bool flag_dist)
        {

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Partition...";

            PQueue pqueue;
            
            // init seed points
            for(int label_generator = 0; label_generator < generators.size(); label_generator++)
            {
                Generator& generator = generators[label_generator];
                int generator_point_index = generator.point_index();
                m_vlabels[generator_point_index] = label_generator;
                generator.qem() = m_vqems[generator_point_index];
                add_candidates(pqueue, generator_point_index, label_generator, flag_dist, m_vlabels, generator);
            }

            // partitioning via region growing
            while(!pqueue.empty())
            {
                const CCandidate candidate = pqueue.top();
                pqueue.pop();
                const int point_index = candidate.handle();
                const int label_generator = candidate.index();

                // skip if point already partitioned
                if(m_vlabels.find(point_index) != m_vlabels.end())
                    continue;

                // set label
                m_vlabels[point_index] = label_generator;
                Generator& generator = generators[label_generator];
                generator.add_qem(m_vqems[point_index]); // add the point's diffused qem to generator.qem(), the qem of the entire cluster 
                add_candidates(pqueue, point_index, label_generator, flag_dist, m_vlabels, generator);
            }

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "done" << std::endl;
        }

        /// @brief Add the generators candidates to the priority queue
        /// @param pqueue priority queue of candidate points 
        /// @param point_index index of the current point
        /// @param label_generator index of the generator associated to the point
        /// @param flag_dist 
        /// @param m_vlabels 
        /// @param generator 
        void add_candidates(PQueue &pqueue,
        const int point_index,
        const int label_generator,
        const bool flag_dist,
        const std::map<int, int>& m_vlabels,
        Generator& generator)
        {
            for(const int neighbor_index : m_graph[point_index])
            {
                if(m_vlabels.find(neighbor_index) == m_vlabels.end() ) // not assigned a generator label, hence not already partitioned
                {
                    const double error = compute_growing_error(neighbor_index, generator, flag_dist);
                    pqueue.push(CCandidate(neighbor_index, label_generator, error));
                    // handle is the point index
                    // index is the label of generator for whose cluster the point is the candidate
                    // loss is the growing error
                }
            }
        }

        /// @brief Compute the growing error using the qem cost and weighted by the euclidean distance
        /// @param index index of the current point
        /// @param label index of the generator associated to the point
        /// @param flag flag to use the euclidean distance
        /// @param generators 
        /// @return the cost
        double compute_growing_error(const int neighbor_index,
            Generator& generator,
            const bool flag)
        {
            const double qem_cost = compute_qem_error(m_vqems[neighbor_index], generator.location());
            double total_cost = qem_cost; 

            if(flag)
            {
                Point& neighbor_location = m_point_set.point(neighbor_index);
                total_cost += m_dist_weight * CGAL::squared_distance(generator.location(), neighbor_location);
            }
            return total_cost;
        }

        /// @brief Compute the QEM error from a query point 
        /// @param qem 
        /// @param point 
        /// @return the qem error
        double compute_qem_error(QEM_metric& qem, const Point& point)
        {
            Eigen::VectorXd vec(4);
            vec << point.x(), point.y(), point.z(), 1.0;

            const double error = vec.transpose() * qem.get_4x4_matrix() * vec;
            assert(error >= 0.0);

            return error; 
        }

        /// @brief Update the generators 
        /// @param m_vlabels 
        /// @param generators 
        /// @return a Boolean : true if generators changed and false otherwise 
        bool update_generators(std::map<int, int>& vlabels,
            std::vector<Generator>& generators, const KNNTree& tree)
        {
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Update generators...";

            // records point-indices of current generators
            std::vector<double> min_qem_errors;
            std::vector<int> old_generators;
            for (int label = 0; label < generators.size(); label++)
            {
                Generator& generator = generators[label];
                old_generators.push_back(generator.point_index());
                min_qem_errors.push_back(1e308); // fixme
            }

            // update generators : optimal qem points for the sum of qem matrices in each cluster
            for (int point_index = 0; point_index < m_point_set.size(); point_index++)
            {
                // skip points not labelled
                if (vlabels.find(point_index) == vlabels.end())
                    continue;

                // get qem of point's cluster 
                int label = vlabels[point_index];
                Generator& generator = generators[label];

                // compute QEM optimal point of generator's cluster, with current point as seed
                Point& seed = m_point_set.point(point_index);
                Point optimal_location= compute_optimal_point(generator.qem(), seed);

                // compute QEM error
                const double qem_error = compute_qem_error(generator.qem(), optimal_location);

                if (qem_error < min_qem_errors[label])
                    {
                        generator.point_index() = point_index;
                        generator.location() = optimal_location;
                        min_qem_errors[label] = qem_error;
                    }
            }

            // compute the nearest point from the optimal locations of all generators
            // experimental step!!

            for (int label = 0; label < generators.size(); label++) {
                Generator& generator = generators[label];
                Point generator_point = generator.location();

                K_neighbor_search search(tree, generator_point, 1);

                for (KNNIterator it = search.begin(); it != search.end(); it++)
                {
                    generator.point_index() = (it->first).second;
                    generator.location() = m_point_set.point(generator.point_index());
                }
            }

            if(m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "done" << std::endl;

            // check changes of generators
            for (int i = 0; i < generators.size(); i++)
                if (generators[i].point_index() != old_generators[i])
                    return true;

            // generators have not changed
            return false; 
        }

        // compute clustering errors (total, max, average, variance, etc)
        // @param m_vlabels
        // @param m_generators
        // returns a pair {total error , variance}
        std::pair<double,double> compute_errors(std::map<int, int>& vlabels,
            std::vector<Generator>& generators)
        {
            double max_error = 0.0;
            double sum_errors = 0.0;
            std::vector<double> sum_cluster_errors(generators.size(), 0.0);

            // compute errors

            for (int point_index = 0; point_index < m_point_set.size(); point_index++)
            {
                // skip unlabelled point
                if (vlabels.find(point_index) == vlabels.end())
                    continue;

                // get generator
                int label = vlabels[point_index];
                Generator& generator = generators[label];

                // compute QEM error
                const double error = compute_qem_error(m_vqems[point_index], generator.location());

                sum_errors += error;
                sum_cluster_errors[label] += error;
                max_error = error > max_error ? error : max_error;
            }

            double average = sum_errors / (double)generators.size();
            double variance = 0.0;
            for(int label = 0; label < sum_cluster_errors.size(); label++)
            {
                const double diff = sum_cluster_errors[label] - average;
                variance += average * average;
            }

            std::cout << "Clustering errors: ";
            std::cout << "Total: " << sum_errors << '\t';
            std::cout << "Average: " << average << '\t';
            std::cout << "Variance: " << variance << std::endl;

            return std::make_pair (sum_errors,variance);
        }


        /// @brief Compute optimal point using either SVD or the direct inverse
        /// @param cluster_qem 
        /// @param cluster_pole 
        /// @return the optimal point
        Point compute_optimal_point(QEM_metric& cluster_qem, Point& cluster_pole)
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

                optim(0) = cluster_pole.x();
                optim(1) = cluster_pole.y();
                optim(2) = cluster_pole.z();
                optim(3) = 1.;

                optim = optim + svd_decomp.solve(qem_vec - qem_mat * optim);
            }

            Point optim_point(optim(0), optim(1), optim(2));

            return optim_point;
        }

        /// @brief Splits the cluster if the qem error is more than a threshold split_thresh
        /// @param m_vlabels 
        /// @param generators 
        /// @param m_diag diagonal of the aabb box
        /// @param m_spacing average spacing between the points 
        /// @param split_ratio user defined parameter for the split
        /// @return total of generators added
        size_t guided_split_clusters(
        std::map<int, int>& vlabels,
        std::vector<Generator>& generators,
        const double diag,
        const double spacing,
        const double split_ratio,
        const int iteration) // batch splitting
        {
            double split_thresh = split_ratio * diag * spacing;
            split_thresh = split_thresh * split_thresh; // square distance
            split_thresh = split_thresh * m_num_knn; 

            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Iteration: " << iteration << std::endl;
                std::cout << "Spitting threshold: " << split_thresh << std::endl;
            }

            std::cout << "generators.size(): " << generators.size() << std::endl;
            DoubleList max_qem_errors(generators.size(), 0.); // maximum QEM error per cluster
            IntList point_indices(generators.size(), -1); // index of the point realising the maximum error

            for(int point_index = 0; point_index < m_point_set.size(); point_index++)
            {
                // skip unlabelled point
                if (vlabels.find(point_index) == vlabels.end()) {
                    continue;
                }

                // get generator and point
                int label_generator = vlabels[point_index];
                Generator& generator = generators[label_generator];
                Point& point = m_point_set.point(point_index);

                // copmute qem error of the point in cluster's qem
                // FIXME: also try with generator's diffused qem
                const double error = compute_qem_error(generator.qem(), point); 
                //const double error = compute_qem_error(m_vqems[generator.point_index()], point);

                if(error > max_qem_errors[label_generator])
                {
                    max_qem_errors[label_generator] = error;
                    point_indices[label_generator] = point_index;
                }
            }

            // points exceeding maximum error threshold
            IntList new_generators; // point indices of new generators
            double max_error = -1e308;
            std::set<Point> generator_locations;
            for (int label_generator = 0; label_generator < generators.size(); label_generator++)
            {
                generator_locations.insert(generators[label_generator].location());
            }

            for(int label_generator = 0; label_generator < generators.size(); label_generator++)
            {  
                max_error = (std::max)(max_qem_errors[label_generator], max_error);
                if (point_indices[label_generator] == -1)   // why is this happening?
                    continue;
                int new_generator_index = point_indices[label_generator];
                Point new_generator_location = m_point_set.point(new_generator_index);
                if (max_qem_errors[label_generator] > split_thresh  // max error of cluster exceeds threshold 
                    && (generator_locations.find(new_generator_location) == generator_locations.end()))  // and the point realising the max error is not a current generator
                {
                    new_generators.push_back(new_generator_index);
                    generator_locations.insert(new_generator_location);
                }
            }
               
            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Maximum error among all clusters: " << max_error << std::endl;
                std::cout << "Found " << new_generators.size() << " new generators!" << std::endl;
            }

            // merge close generators
            std::set<int, std::greater<int>> duplicate_generators;  // indices of duplicate generators in new_generators sorted in descending order
            double dist_thresh = spacing;  // distance threshold for merging clusters
            dist_thresh = m_num_knn * dist_thresh * dist_thresh;
            //double dist_thresh = spacing * spacing;

            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Distance threshold for merging clusters: " << dist_thresh << std::endl;
            }
            double min_generator_dist = 1e308;

            for(int i = 0; i < new_generators.size(); i++)
            {
                for(int j = i + 1; j < new_generators.size(); j++)
                {
                    Point point_i = m_point_set.point(new_generators[i]);
                    Point point_j = m_point_set.point(new_generators[j]);
                    min_generator_dist = (std::min)(CGAL::squared_distance(point_i, point_j), min_generator_dist);

                    if(duplicate_generators.find(j) == duplicate_generators.end() &&  // generator not marked duplicate already
                       CGAL::squared_distance(point_i, point_j) < dist_thresh) // distance between generators is less than the minimum threshold
                    {
                        duplicate_generators.insert(j);
                    }
                }
            }

            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Minimum distance between two new generators: " << min_generator_dist << std::endl;
            }

            // removing duplicate generators from the list of new generators
            for(auto& index: duplicate_generators)
            {
                new_generators.erase(new_generators.begin() + index);
            }

            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Remove " << duplicate_generators.size() << " duplicated generators!" << std::endl;
            }

            // insert new generators
            for(int index = 0; index < new_generators.size(); index++)
            {
                int generator_point_index = new_generators[index];
                generators.push_back(Generator(generator_point_index, m_point_set.point(generator_point_index)));
            }

            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Finally added " << new_generators.size() << " new generators!" << std::endl;
            }

            return new_generators.size();
        }

        /// @brief write some data to csv files and print worst error at each iteration 
        void write_csv()
        {
            csv_writer->writeDataErrorGeneratorsToCSV("error_generators.csv");
            csv_writer->writeDataErrorPointsToCSV("error_points.csv");
            csv_writer->printWorst();
        }

        /// @brief set distance weight (lambda coefficient in region growing)
        /// @param dist_weight distance weight
        void set_distance_weight(double dist_weight)
        {
            m_dist_weight = dist_weight;
        }

        /// @brief number of connected components
        /// returns the number of connected components in the KNN graph
        int get_component_count()
        {
            return component_count;
        }

        /// @brief vertex QEM
        /// @param index of the vertex in the pointset 
        /// returns the diffused QEM of the point
        QEM_metric& vqem(const int index)
        {
            assert(index > 0);
            assert(index < m_vqems.size());
            return m_vqems[index];
        }

        private:

            Pointset m_point_set;  // Pointset of the point cloud
            unsigned int m_num_knn = 12;    // number of nearest neighbours in k nearest neighbours
            double m_dist_weight = 0.1;   // lambda term in region growing

            // qem
            std::vector<QEM_metric> m_pqems; // qem per point
            std::vector<QEM_metric> m_vqems; // diffused qem per point
            std::vector<std::vector<int> > m_graph; // neighborhood graph (indices)
            std::vector<bool> m_visited;
            std::vector<int> m_component;
            int component_count = 0; // fixme

            VERBOSE_LEVEL m_verbose_level = qem::VERBOSE_LEVEL::HIGH;

            // csv
            std::shared_ptr<DataWriter> csv_writer;
            
    };
}