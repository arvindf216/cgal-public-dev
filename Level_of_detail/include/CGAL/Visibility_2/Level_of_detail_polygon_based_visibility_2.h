#ifndef CGAL_LEVEL_OF_DETAIL_POLYGON_BASED_VISIBILITY_2_H
#define CGAL_LEVEL_OF_DETAIL_POLYGON_BASED_VISIBILITY_2_H

#if defined(WIN32) || defined(_WIN32) 
#define PSR "\\" 
#else 
#define PSR "/" 
#endif 

// STL includes.
#include <utility>
#include <cassert>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>

// Boost includes.
#include <boost/tuple/tuple.hpp>

// CGAL includes.
#include <CGAL/utils.h>
#include <CGAL/IO/Color.h>
#include <CGAL/number_utils.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/property_map.h>

// New CGAL includes.
#include <CGAL/Mylog/Mylog.h>

namespace CGAL {

	namespace LOD {

		template<class KernelTraits, class InputContainer, class DataStructure>
		class Level_of_detail_polygon_based_visibility_2 {

        public:
            typedef KernelTraits   Kernel;
            typedef InputContainer Input;
            typedef DataStructure  Data_structure;

            typename Kernel::Compute_squared_distance_2 squared_distance;

            using FT  = typename Kernel::FT;
            using Log = CGAL::LOD::Mylog;

            using Point_2 = typename Kernel::Point_2;
            using Point_3 = typename Kernel::Point_3;
            
            using Line_2 = typename Kernel::Line_2;

            using Point_label     = int;
            using Point_label_map = typename Input:: template Property_map<Point_label>;

			using Point_with_label = typename std::pair<Point_2, Point_label>;
			using Point_map        = typename CGAL::First_of_pair_property_map<Point_with_label>;
            using Points           = std::vector<Point_with_label>;

			using Search_traits_2 = CGAL::Search_traits_2<Kernel>;
			using Search_traits   = CGAL::Search_traits_adapter<Point_with_label, Point_map, Search_traits_2>;
			using Fuzzy_circle    = CGAL::Fuzzy_sphere<Search_traits>;
			using Fuzzy_tree      = CGAL::Kd_tree<Search_traits>;

            using Container  = typename Data_structure::Container;
            using Containers = typename Data_structure::Containers;
				
			using Polygon                 = typename Container::Polygon;
			using Polygon_vertex_iterator = typename Polygon::Vertex_const_iterator;

            using Visibility_data = std::pair<FT, FT>;
            using Visibility      = std::map<size_t, Visibility_data>;

            using Point_iterator = typename Input::const_iterator;
            using Colour         = typename Container::Colour;

            Level_of_detail_polygon_based_visibility_2(const Input &input, Data_structure &data_structure) :
            m_silent(false), m_input(input), m_data_structure(data_structure) { 
                
                set_point_labels();
            }

            void compute() {

                Visibility visibility;

                set_points();
                set_initial_visibility(visibility);

                // estimate_visibility(visibility); // add or remove if necessary
                set_visibility_to_containers(visibility);

                if (!m_silent) save_data_structure();
            }

            void make_silent(const bool new_state) {
                m_silent = new_state;
            }

        private:
            bool m_silent;
            Point_label_map m_point_labels;

            const Input    &m_input;
            Data_structure &m_data_structure;

            Points m_points;

			inline void set_point_labels() {
				boost::tie(m_point_labels, boost::tuples::ignore) = m_input.template property_map<Point_label>("label");
			}

            void set_points() {
                m_points.clear();
                m_points.resize(m_input.number_of_points());

                Point_2 point; size_t i = 0;
                for (Point_iterator pit = m_input.begin(); pit != m_input.end(); ++pit, ++i) {

                    const Point_3 &original = m_input.point(*pit);
                    point = Point_2(original.x(), original.y());

                    m_points[i] = std::make_pair(point, get_point_label(pit));
                }
            }

            Point_label get_point_label(const Point_iterator &pit) {
                assert(m_point_labels.size() == m_input.number_of_points());
                return m_point_labels[*pit];
            }

            void set_initial_visibility(Visibility &visibility) {
                
                const Containers &containers = m_data_structure.containers();
				assert(containers.size() > 0);

                assert(m_points.size() > 0);
                Fuzzy_tree tree(m_points.begin(), m_points.end());

                visibility.clear();
                for (size_t i = 0; i < containers.size(); ++i)
                    visibility[i] = get_initial_visibility(tree, i);
            }

            Visibility_data get_initial_visibility(const Fuzzy_tree &tree, const size_t container_index) {
                
                // Old method.
                // return std::make_pair(FT(0), FT(0));

                // New method.
                const Containers &containers = m_data_structure.containers();
                return get_barycentre_visibility(tree, containers[container_index].polygon);
            }

            Visibility_data get_barycentre_visibility(const Fuzzy_tree &tree, const Polygon &polygon) {

                const size_t num_vertices = std::distance(polygon.vertices_begin(), polygon.vertices_end());
                std::vector<Point_2> vertices(num_vertices);

                FT x = FT(0), y = FT(0); size_t i = 0;
                for (Polygon_vertex_iterator vit = polygon.vertices_begin(); vit != polygon.vertices_end(); ++vit, ++i) {
                    const Point_2 &vertex = *vit;

                    x += vertex.x();
                    y += vertex.y();

                    vertices[i] = Point_2(vertex.x(), vertex.y());
                }
                assert(num_vertices > 0);

                x /= static_cast<FT>(num_vertices);
                y /= static_cast<FT>(num_vertices);

                const Point_2 barycentre(x, y);
                FT min_dist = FT(1000000000000);

                for (i = 0; i < num_vertices; ++i) {
                    const size_t ip = (i + 1) % num_vertices;

                    const Point_2 &p1 = vertices[i];
                    const Point_2 &p2 = vertices[ip];

                    const Line_2 line(p1, p2);
                    const Point_2 projected = line.projection(barycentre);

                    const FT dist = static_cast<FT>(CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(barycentre, projected))));
                    min_dist = CGAL::min(min_dist, dist);
                }

                assert(min_dist > FT(0));

                const FT radius = min_dist * FT(1.2);
                const Fuzzy_circle circle(barycentre, radius);

                Points found_points;
				tree.search(std::back_inserter(found_points), circle);

                const size_t container_index = 0;

                Visibility visibility;
                visibility[container_index] = std::make_pair(FT(0), FT(0));

                assert(found_points.size() > 0);
                for (i = 0; i < found_points.size(); ++i) {

                    const Point_label found_point_label = found_points[i].second;
                    add_visibility(container_index, found_point_label, visibility);
                }

                return visibility.at(container_index);
            }

            void estimate_visibility(Visibility &visibility) {

                assert(m_points.size() > 0);
				for (size_t i = 0; i < m_points.size(); ++i) {
                    const Point_2 &point = m_points[i].first;

                    const int container_index = m_data_structure.locate(point);
                    if (container_index < 0) continue; 

                    const Point_label point_label = m_points[i].second;
                    add_visibility(static_cast<size_t>(container_index), point_label, visibility);
                }
            }

            void add_visibility(const size_t container_index, const Point_label point_label, Visibility &visibility) {

				const Point_label ground     = 0;
				const Point_label facade     = 1;
				const Point_label roof       = 2;
				const Point_label vegetation = 3;

				switch (point_label) {

					case ground:
						set_outside(container_index, visibility);
						break;

					case facade:
						set_unknown(container_index, visibility);
						break;

					case roof:
						set_inside(container_index, visibility);
						break;

					case vegetation:
						set_outside(container_index, visibility);
						break;

					default:
                        set_unknown(container_index, visibility);
                        break;
				}
			}

            inline void set_inside(const size_t container_index, Visibility &visibility) {
				visibility[container_index].first  += FT(1);
			}

			inline void set_outside(const size_t container_index, Visibility &visibility) {
				visibility[container_index].second += FT(1);
			}

			void set_unknown(const size_t container_index, Visibility &visibility) {
				visibility[container_index].first  += FT(1) / FT(2);
                visibility[container_index].second += FT(1) / FT(2);
			}

            void set_visibility_to_containers(const Visibility &visibility) {

                Containers &containers = m_data_structure.containers();
				assert(containers.size() > 0);

                assert(visibility.size() == containers.size());
                for (size_t i = 0; i < containers.size(); ++i) {

                    containers[i].inside = get_final_visibility_value(visibility.at(i));
                    containers[i].colour = get_final_visibility_colour(containers[i].inside);
                }
            }

            FT get_final_visibility_value(const Visibility_data &visibility_data) {

                const FT inside  = visibility_data.first;
                const FT outside = visibility_data.second;

                if (inside > outside) return FT(1);
                return FT(0);
            }

            Colour get_final_visibility_colour(const FT visibility_value) {
                assert(visibility_value == FT(0) || visibility_value == FT(1));

				if (visibility_value == FT(1)) return Colour(55, 255, 55);
                return Colour(255, 55, 55);
            }

            void save_data_structure() {
                
                const Containers &containers = m_data_structure.containers();
                assert(containers.size() > 0);

                Log exporter;
                exporter.save_polygons<Containers, Polygon, Kernel>(containers, "tmp" + std::string(PSR) + "visibility", true);
            }
        };
    }
}

#endif // CGAL_LEVEL_OF_DETAIL_POLYGON_BASED_VISIBILITY_2_H