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
typedef qem::MergeCandidate MCandidate;
typedef qem::Candidate_more<MCandidate> MMore;
typedef qem::Custom_priority_queue<MCandidate, MMore> MPQueue;

typedef typename qem::CGenerator<Kernel> Generator;

namespace qem
{
    class Clustering_ftc
    {
        private:
            Pointset m_pointset;  // pointset of the point cloud
            unsigned int m_num_knn = 12;    // number of nearest neighbours in k nearest neighbours
            unsigned int m_nb_clusters = 1e3;  // number of final clusters
            MPQueue m_pqueue;     // priority queue for storing merge operators

            // qem
            std::vector<QEM_metric> m_pqems; // qem per point
            std::vector<QEM_metric> m_vqems; // diffused qem per point
            std::vector<std::vector<int> > m_graph; // neighborhood graph (indices)

            VERBOSE_LEVEL m_verbose_level = qem::VERBOSE_LEVEL::HIGH;


    public:

        Clustering_ftc()
        {
        }

        Clustering_ftc(const Pointset& pointset,
            unsigned int num_knn,
            unsigned int nb_clusters,
            VERBOSE_LEVEL verbose_level)
        {
            m_pointset = pointset;
            m_num_knn = num_knn;
            m_nb_clusters = nb_clusters;
            m_verbose_level = verbose_level;
        }

        /// @brief Compute the qem for each point based on the k nearest neighbor neighbors
        // fixme: rather based on the normal and average distance to neighbors!
        // TODO: add function to estimate normals
        void initialize_qem_per_point(const KNNTree& tree)
        {
            // init vector of qems
            m_pqems.clear();

            for (int point_index = 0; point_index < m_pointset.size(); point_index++)
            {
                Point point = m_pointset.point(point_index);
                K_neighbor_search search(tree, point, m_num_knn);
                KNNDistance tr_dist;  // orthogonal distance : transformed distace in k nearest neighbours graph

                double avg_dist = 0.;
                for (KNNIterator it = search.begin(); it != search.end(); it++)
                    avg_dist += tr_dist.inverse_of_transformed_distance(it->second);
                avg_dist = avg_dist / (double)m_num_knn;

                //if (!m_pointset.has_normal_map()) {
                //    m_pointset.add_normal_map();
                //    CGAL::jet_estimate_normals<CGAL::Sequential_tag>
                //        (m_pointset,
                //        m_num_knn, 
                //        m_pointset.parameters(). // Named parameters provided by Point_set_3
                //        degree_fitting(2));     // additional named parameter specific to jet_estimate_normals
                //}

                QEM_metric pqem = compute_qem_for_point(point, m_pointset.normal(point_index), avg_dist * avg_dist / 2);
                m_pqems.push_back(pqem);
            }
        }

        /// @brief Compute the qem for a point weighted by the area of the face
        /// @param query the point to compute the qem for
        /// @param normal of the face
        /// @param area of the face
        /// @return the qem computed
        QEM_metric compute_qem_for_point(const Point& query, const Vector& normal, const double& area)
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
            for (int point_index = 0; point_index < m_pointset.size(); point_index++)
            {
                K_neighbor_search search(m_tree, m_pointset.point(point_index), m_num_knn);

                IntList neighbors;

                // init with qem of point itself 
                QEM_metric vqem = m_pqems[point_index];

                for (KNNIterator it = search.begin(); it != search.end(); it++)
                {
                    auto neighbor_index = (it->first).second; // it->first is of the type Point_with_index
                    vqem = vqem + m_pqems[neighbor_index];
                    neighbors.push_back(neighbor_index);
                }
                m_graph.push_back(neighbors);

                m_vqems.push_back(vqem);
            }
        }

        /// @brief Initialize priority with each pair of points in the kNN graph of the point set
        /// @param vlabels map of point index to generator label
        /// @param pindices vector of point indices corresponding to each generator label
        /// @param generators vector of generators
        void init_pqueue(std::map<int, int>& vlabels,
            std::vector< std::vector<int> >& pindices,
            std::vector<Generator>& generators) {
            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Initializing priority queue..." << std::endl;

            for (int label_generator = 0; label_generator < generators.size(); label_generator++) {
                // current generator
                Generator& generator = generators[label_generator];
                int generator_point_index = generator.point_index();                                // pointset index of the point in generator's cluster

                // neigbors of the generator
                for (int neighbor_point_index : m_graph[generator_point_index]) 
                {                                                                                   // point set index of the KNN graph neighbor of the cluster point
                    int label_neighbor_generator = vlabels[neighbor_point_index];                   // generator label of the neighbor point's cluster

                    // adding in priority queue
                    double delta_error = compute_delta_error(generators, pindices, label_generator, neighbor_point_index);
                    MCandidate candidate = MCandidate(label_generator, label_neighbor_generator, delta_error);
                    if (!m_pqueue.contains(candidate))
                        m_pqueue.push(candidate);
                }
            }

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
            {
                std::cout << "Smallest delta error: " << m_pqueue.top().loss() << std::endl;
                std::cout << "Priority queue initialized with: "<< m_pqueue.size() << " merge operators" << std::endl;
            }
        }

        /// @brief Delta error for merging the clusters corresponding to the given generator labels
        /// @param generators vector of generators
        /// @param pindices vector of point indices corresponding to each generator label
        /// @param label_generator label of the generator
        /// @param label_neighbor_generator label of the neigboring generator
        /// return the delta error
        double compute_delta_error(std::vector<Generator>& generators,
            std::vector< std::vector<int> >& pindices,
            int label_generator,
            int label_neighbor_generator)
        {
            // generator
            Generator& generator = generators[label_generator];
            Point generator_location = generator.location();                                    // location of the generator
            QEM_metric& point_qem = generator.qem();                                            // sum of diffused qems of the points in generator's cluster 
            const double generator_error = compute_qem_error(point_qem, generator_location);    // total error of generator's cluster
            
            // neighbor generator
            Generator& neighbor_generator = generators[label_neighbor_generator];
            Point neighbor_generator_location = neighbor_generator.location();              // location of the neighbor generator
            QEM_metric& neighbor_qem = neighbor_generator.qem();                            // sum of diffused qems of the points in neighbor generator's cluster
            const double neighbor_generator_error = compute_qem_error(neighbor_qem, neighbor_generator_location);   // total error of neighbor generator's cluster

            // merging
            double initial_error = generator_error + neighbor_generator_error;              // total error before merging
            QEM_metric final_qem = point_qem + neighbor_qem;                                // final qem after merging
            Point final_optimal_location = compute_optimal_point(final_qem, generator_location);     // final optimal location after merging

            //int nearest_point_index = -1;
            //double nearest_distance = 1e308;

            //for (int point_index : pindices[label_generator])
            //{
            //    Point point = m_pointset.point(point_index);
            //    if (CGAL::squared_distance(point, final_optimal_location) < nearest_distance) {
            //        nearest_point_index = point_index;
            //        nearest_distance = CGAL::squared_distance(point, final_optimal_location);
            //    }
            //}

            //for (int point_index : pindices[label_neighbor_generator])
            //{
            //    Point point = m_pointset.point(point_index);
            //    if (CGAL::squared_distance(point, final_optimal_location) < nearest_distance) {
            //        nearest_point_index = point_index;
            //        nearest_distance = CGAL::squared_distance(point, final_optimal_location);
            //    }
            //}

            //if (nearest_point_index != -1) {
            //    final_optimal_location = m_pointset.point(nearest_point_index);
            //}

            double final_error = compute_qem_error(final_qem, final_optimal_location);      // total error after merging

            double delta_error = final_error - initial_error;
            return delta_error;
        }

        /// @brief Clustering: perform single linkage clustering till number of clusters reaches m_nb_clusters,
        /// the number of clusters provided by the user
        /// @param vlabels map of point index to generator label
        /// @param pindices vector of point indices corresponding to each generator label
        /// @param generators vector of generators
        void clustering(std::map<int, int>& vlabels,
            std::vector< std::vector<int> >& pindices,
            std::vector<Generator>& generators)
        {

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Clustering..." << std::endl;

            int num_clusters = generators.size();

            // clustering via merging nearby clusters
            while (!m_pqueue.empty() && num_clusters > m_nb_clusters)
            {

                std::cout << "Number of clusters: " << num_clusters << std::endl;

                const MCandidate candidate = m_pqueue.top();
                m_pqueue.pop();
                int first_generator_label = candidate.label_first();
                int second_generator_label = candidate.label_second();
                
                // removing all the generators neighboring the "clusters to be merged" from the priority queue
                std::vector<MCandidate> removed_candidates = m_pqueue.remove_all_ftc(candidate);
                std::set<int> neighbor_labels;

                for (MCandidate removed_candidate : removed_candidates) {     
                    // insert into the set labels of all the generators of the clusters neighboring the pair of clusters being merged 
                    neighbor_labels.insert(removed_candidate.label_first());
                    neighbor_labels.insert(removed_candidate.label_second());
                }

                // remove the labels of the generators of the pair of clusters to be merged
                neighbor_labels.erase(first_generator_label);
                neighbor_labels.erase(second_generator_label);

                // merge the clusters
                Generator& first_generator = generators[first_generator_label];
                Generator& second_generator = generators[second_generator_label];

                if (!first_generator.is_active())
                    continue;

                first_generator.add_qem(second_generator.qem());  // adding the second generator's qem to the first generators'
                Point seed = first_generator.location();
                first_generator.location() = compute_optimal_point(first_generator.qem(), seed);
                second_generator.is_active() = false;               // making the second generator inactive

                for (int point_index : pindices[second_generator_label])
                {
                    // changing the label of the generators which had the label of the second generator
                    vlabels[point_index] = first_generator_label;
                }

                // updating indices of points in first generator's cluster
                std::vector<int> second_cluster_points;
                second_cluster_points.assign(pindices[second_generator_label].begin(), pindices[second_generator_label].end());
                pindices[first_generator_label].insert(pindices[first_generator_label].end(), second_cluster_points.begin(), second_cluster_points.end());
                pindices[second_generator_label].clear();

                // updating first_generator.point_index() to the index of the nearest pointcloud point
                // in the generator's cluster
                int nearest_point_index = -1;
                double nearest_point_distance = 1e308;

                for (int point_index : pindices[first_generator_label]) {
                    double current_point_distance = CGAL::squared_distance(m_pointset.point(point_index), first_generator.location());

                    if (current_point_distance < nearest_point_distance) {
                        nearest_point_distance = current_point_distance;
                        nearest_point_index = point_index;
                    }
                }

                if (nearest_point_index != -1) {
                    first_generator.point_index() = nearest_point_index;
                    //first_generator.location() = m_pointset.point(nearest_point_index);
                }

                // reinserting all the neighbor generators with updated error values 
                for (std::set<int>::iterator it = neighbor_labels.begin(); it != neighbor_labels.end(); ++it) 
                {
                    int add_generator_label = *it;
                    double delta_error = compute_delta_error(generators, pindices, first_generator_label, add_generator_label);
                    MCandidate candidate = MCandidate(first_generator_label, add_generator_label, delta_error);
                    m_pqueue.push(candidate);
                }

                num_clusters--;  // number of clusters decreases by one every iteration
            }

            // removing all the inactive generators 
            std::vector<Generator> new_generators;      // vector of all the active generators
            std::map <int, int> new_labels;             // map of old generator_labels to the new generator_labels

            for (int label_generator = 0; label_generator < generators.size(); label_generator++) {
                Generator generator = generators[label_generator];
                if (generator.is_active()) {
                    new_generators.push_back(generator);
                    new_labels[label_generator] = new_generators.size() - 1;
                }
            }

            // changing the labels of generators in vlabels
            for (int point_index = 0; point_index < m_pointset.size(); point_index++) {
                int old_label_generator = vlabels[point_index];
                vlabels[point_index] = new_labels[old_label_generator];
            }

            generators.clear();
            generators.assign(new_generators.begin(), new_generators.end());

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
            {
                std::cout << "Final number of generators: " << generators.size() << std::endl;
                std::cout << "done" << std::endl;
            }
        }

        /// @brief Clustering: perform single linkage clustering till number of clusters reaches m_nb_clusters,
        /// the number of clusters provided by the user
        /// compute errors every 1000 iterations
        /// save clusterings to ply every 10000 iterations
        /// @param file output file stream for storing errors
        /// @param vlabels map of point index to generator label
        /// @param pindices vector of point indices corresponding to each generator label
        /// @param generators vector of generators
        void clustering_and_compute_errors(std::ofstream& file, std::map<int, int>& vlabels,
            std::vector< std::vector<int> >& pindices,
            std::vector<Generator>& generators)
        {

            if (m_verbose_level != VERBOSE_LEVEL::LOW)
                std::cout << "Clustering..." << std::endl;

            int num_clusters = generators.size();
            int iteration = 0;

            // clustering via merging nearby clusters
            while (!m_pqueue.empty() && num_clusters > m_nb_clusters)
            {

                std::cout << "Number of clusters: " << num_clusters << std::endl;
                if (iteration % 1000 == 0) {
                    const std::pair<double, double> total_error = compute_errors(generators);
                    file << iteration << "," << total_error.first << "," << total_error.second << std::endl;
                }

                if (iteration % 10000 == 0) {
                    int num = iteration / 10000;
                    std::string filename("clustering_ftc-");
                    filename.append(std::to_string(num));
                    filename.append(std::string(".ply"));
                    save_clustering_to_ply(generators, vlabels, filename);
                }

                const MCandidate candidate = m_pqueue.top();
                m_pqueue.pop();
                int first_generator_label = candidate.label_first();
                int second_generator_label = candidate.label_second();

                // removing all the generators neighboring the "clusters to be merged" from the priority queue
                std::vector<MCandidate> removed_candidates = m_pqueue.remove_all_ftc(candidate);
                std::set<int> neighbor_labels;

                for (MCandidate removed_candidate : removed_candidates) {
                    // insert into the set labels of all the generators of the clusters neighboring the pair of clusters being merged 
                    neighbor_labels.insert(removed_candidate.label_first());
                    neighbor_labels.insert(removed_candidate.label_second());
                }

                // remove the labels of the generators of the pair of clusters to be merged
                neighbor_labels.erase(first_generator_label);
                neighbor_labels.erase(second_generator_label);

                // merge the clusters
                Generator& first_generator = generators[first_generator_label];
                Generator& second_generator = generators[second_generator_label];

                if (!first_generator.is_active() || pindices[first_generator_label].empty())
                    continue;

                first_generator.add_qem(second_generator.qem());  // adding the second generator's qem to the first generators'
                Point seed = first_generator.location();
                first_generator.location() = compute_optimal_point(first_generator.qem(), seed);
                second_generator.is_active() = false;               // making the second generator inactive

                for (int point_index : pindices[second_generator_label])
                {
                    // changing the label of the generators which had the label of the second generator
                    vlabels[point_index] = first_generator_label;
                }

                // updating indices of points in first generator's cluster
                std::vector<int> second_cluster_points;
                second_cluster_points.assign(pindices[second_generator_label].begin(), pindices[second_generator_label].end());
                pindices[first_generator_label].insert(pindices[first_generator_label].end(), second_cluster_points.begin(), second_cluster_points.end());
                pindices[second_generator_label].clear();

                // updating first_generator.point_index() to the index of the nearest pointcloud point
                // in the generator's cluster
                int nearest_point_index = -1;
                double nearest_point_distance = 1e308;

                for (int point_index : pindices[first_generator_label])
                {
                    double current_point_distance = CGAL::squared_distance(m_pointset.point(point_index), first_generator.location());

                    if (current_point_distance < nearest_point_distance) {
                        nearest_point_distance = current_point_distance;
                        nearest_point_index = point_index;
                    }
                }

                if (nearest_point_index != -1) {
                    first_generator.point_index() = nearest_point_index;
                    //first_generator.location() = m_pointset.point(nearest_point_index);
                }

                // reinserting all the neighbor generators with updated error values in the priority queue 
                for (std::set<int>::iterator it = neighbor_labels.begin(); it != neighbor_labels.end(); ++it)
                {
                    int add_generator_label = *it;
                    double delta_error = compute_delta_error(generators, pindices, first_generator_label, add_generator_label);
                    MCandidate candidate = MCandidate(first_generator_label, add_generator_label, delta_error);
                    m_pqueue.push(candidate);
                }

                num_clusters--;  // number of clusters decreases by one every iteration
                iteration++;
            }

            // removing all the inactive generators 
            std::vector<Generator> new_generators;      // vector of all the active generators
            std::map <int, int> new_labels;             // map of old generator_labels to the new generator_labels

            for (int label_generator = 0; label_generator < generators.size(); label_generator++) {
                Generator generator = generators[label_generator];
                if (generator.is_active()) {
                    new_generators.push_back(generator);
                    new_labels[label_generator] = new_generators.size() - 1;
                }
            }

            // changing the labels of generators in vlabels
            for (int point_index = 0; point_index < m_pointset.size(); point_index++) {
                int old_label_generator = vlabels[point_index];
                vlabels[point_index] = new_labels[old_label_generator];
            }

            generators.clear();
            generators.assign(new_generators.begin(), new_generators.end());

            if (m_verbose_level != VERBOSE_LEVEL::LOW) {
                std::cout << "Final number of generators: " << generators.size() << std::endl;
                std::cout << "done" << std::endl;
            }

        }


        // compute clustering errors (total, max, average, variance, etc)
        // @param m_vlabels
        // @param m_generators
        // returns a pair {total error , variance}
        std::pair<double, double> compute_errors(std::vector<Generator>& generators)
        {
            double max_error = 0.0;
            double sum_errors = 0.0;
            std::vector<double> sum_cluster_errors(generators.size(), 0.0);
            double num_generators = 0.0;

            // compute errors

            for (int label_generator = 0; label_generator < generators.size(); label_generator++)
            {
                Generator& generator = generators[label_generator];

                // skip inactive generators
                if (!generator.is_active())
                    continue;

                num_generators++;

                // compute QEM error
                const double error = compute_qem_error(generator.qem(), generator.location());

                sum_errors += error;
                sum_cluster_errors[label_generator] += error;
                max_error = error > max_error ? error : max_error;
            }

            //for (int point_index = 0; point_index < m_pointset.size(); point_index++)
            //{
            //    // skip unlabelled point
            //    if (vlabels.find(point_index) == vlabels.end())
            //        continue;

            //    // get generator
            //    int label = vlabels[point_index];
            //    Generator& generator = generators[label];

            //    if (!generator.is_active())
            //        continue;

            //    // compute QEM error
            //    const double error = compute_qem_error(m_vqems[point_index], generator.location());

            //    sum_errors += error;
            //    sum_cluster_errors[label] += error;
            //    max_error = error > max_error ? error : max_error;
            //}

            double average = sum_errors / num_generators;
            double variance = 0.0;
            for (int label = 0; label < sum_cluster_errors.size(); label++)
            {
                const double diff = sum_cluster_errors[label] - average;
                variance += average * average;
            }

            std::cout << "Clustering errors: ";
            std::cout << "Total: " << sum_errors << '\t';
            std::cout << "Average: " << average << '\t';
            std::cout << "Variance: " << variance << std::endl;

            return std::make_pair(sum_errors, variance);
        }

        /// Save clusterings to ply using different colors
        /// @param generators vector of generators
        /// @param vlabels map of point index to generator label
        /// @param filename file name of .ply file
        void save_clustering_to_ply(std::vector<Generator>& generators,
            std::map<int,int>& vlabels,
            std::string& filename)
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
            for (int i = 0; i < generators.size(); i++)
            {
                double r = (double)rand() / (RAND_MAX);
                double g = (double)rand() / (RAND_MAX);
                double b = (double)rand() / (RAND_MAX);
                colors.push_back(Vector(r, g, b));
            }

            for (int point_index = 0; point_index < m_pointset.size(); point_index++)
            {
                if (vlabels.find(point_index) == vlabels.end())
                    continue;

                Vector& color = colors[vlabels[point_index]];

                Point point = m_pointset.point(point_index);
                file << point.x() << " " << point.y() << " " << point.z() << " ";
                int r = static_cast<int>(255 * color.x());
                int g = static_cast<int>(255 * color.y());
                int b = static_cast<int>(255 * color.z());
                file << r << " " << g << " " << b << "\n";
            }

            file.close();
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
            if (lu_decomp.isInvertible())
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

        /// @brief vertex QEM
        /// @param index of the vertex in the pointset 
        /// returns the diffused QEM of the point
        QEM_metric& vqem(const int index)
        {
            assert(index > 0);
            assert(index < m_vqems.size());
            return m_vqems[index];
        }

    };
}