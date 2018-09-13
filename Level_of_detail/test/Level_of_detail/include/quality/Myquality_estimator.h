#ifndef CGAL_LEVEL_OF_DETAIL_MYQUALITY_ESTIMATOR_H
#define CGAL_LEVEL_OF_DETAIL_MYQUALITY_ESTIMATOR_H

// STL includes.
#include <map>
#include <string>
#include <cstdlib>
#include <iostream>

// LOD includes.
/*
#include <CGAL/Level_of_detail/Enumerations.h>
#include <CGAL/Level_of_detail/Level_of_detail.h> */

// Local includes.
#include "../terminal/Myterminal_parser.h"

namespace CGAL {

	namespace Level_of_detail {

		namespace LOD = CGAL::Level_of_detail;

		template<class InputKernel>
		class Myquality_estimator {

		public:
			using Kernel = InputKernel;
			using FT 	 = typename Kernel::FT;

			using Lod_parameters = LOD::Myterminal_parser<FT>;
			
			/*
			using Lod_base = LOD::Level_of_detail<Kernel>;
			using Params   = char**;

			using Data = std::vector<FT>;
			using Lod_complexity = typename Lod_base::Lod_complexity;
			using Lod_distortion = typename Lod_base::Lod_distortion;
			using Lod_coverage   = typename Lod_base::Lod_coverage;

			typedef typename Lod_parameters::Input_parameters Parameters; */
			
			Myquality_estimator(const int num_params, const Params params) : 
			m_left_iters(10), 
			m_right_iters(10),
			m_left_bound(FT(1)), 
			m_right_bound(FT(10)) {

				/*
				Lod_parameters lod_parameters(num_params, params);

				m_lod_base.set_optimal_configuration();
				m_lod_base.set_user_defined_parameters(lod_parameters);

				m_lod_base.create_lods();
				set_quality_parameters(lod_parameters); */
			}

			/*
			void compute_data() {
				
				clear();
				const FT initial_x = m_lod_base.get_scale();
				
				compute_x_data(initial_x);
				compute_y_data();
			}

			const Data &retreive_x_data() const {
				
				CGAL_precondition(!m_x_data.empty());
				return m_x_data;
			}

			const Data &retreive_cmp_y_data(const Quality_data_type quality_data_type) const {
				switch (quality_data_type) {

					case Quality_data_type::CMP_ROOFS:
						CGAL_precondition(!m_cmp_roofs_y_data.empty());
						return m_cmp_roofs_y_data;

					case Quality_data_type::CMP_WALLS:
						CGAL_precondition(!m_cmp_walls_y_data.empty());
						return m_cmp_walls_y_data;

					case Quality_data_type::CMP:
						CGAL_precondition(!m_cmp_total_y_data.empty());
						return m_cmp_total_y_data;

					default:
						CGAL_precondition(!"Wrong quality data type!"); exit(EXIT_FAILURE);
						return m_cmp_total_y_data;
				}
			}

			const std::vector<Data> &retreive_dst_y_data(const Quality_data_type quality_data_type) const {
				switch (quality_data_type) {

					case Quality_data_type::DST_ROOFS:
						CGAL_precondition(!m_dst_roofs_y_data.empty());
						return m_dst_roofs_y_data;

					case Quality_data_type::DST_WALLS:
						CGAL_precondition(!m_dst_walls_y_data.empty());
						return m_dst_walls_y_data;

					default:
						CGAL_precondition(!"Wrong quality data type!"); exit(EXIT_FAILURE);
						return m_dst_roofs_y_data;
				}
			}

			const Data &retreive_cov_y_data(const Quality_data_type quality_data_type) const {
				switch (quality_data_type) {

					case Quality_data_type::COV_ROOFS:
						CGAL_precondition(!m_cov_roofs_y_data.empty());
						return m_cov_roofs_y_data;

					case Quality_data_type::COV_WALLS:
						CGAL_precondition(!m_cov_walls_y_data.empty());
						return m_cov_walls_y_data;

					case Quality_data_type::COV:
						CGAL_precondition(!m_cov_total_y_data.empty());
						return m_cov_total_y_data;

					default:
						CGAL_precondition(!"Wrong quality data type!"); exit(EXIT_FAILURE);
						return m_cov_total_y_data;
				}
			} */

		private:
			size_t m_left_iters;
			size_t m_right_iters;

			FT m_left_bound;
			FT m_right_bound;

			/*
			Lod_base m_lod_base;

			Data m_x_data;
			Data m_cmp_roofs_y_data, m_cmp_walls_y_data, m_cmp_total_y_data;
			std::vector<Data> m_dst_roofs_y_data, m_dst_walls_y_data;
			Data m_cov_roofs_y_data, m_cov_walls_y_data, m_cov_total_y_data;
			
			std::shared_ptr<Lod_complexity> m_lod_complexity;
			std::shared_ptr<Lod_distortion> m_lod_distortion;
			std::shared_ptr<Lod_coverage>   m_lod_coverage; */

			/*
			void set_quality_parameters(const Lod_parameters &lod_parameters) {
				const Parameters &parameters = lod_parameters.get();

				add_scalar_parameter("-left_iters" , m_left_iters , parameters);
				add_scalar_parameter("-right_iters", m_right_iters, parameters);

				add_scalar_parameter("-left_bound" , m_left_bound , parameters);
				add_scalar_parameter("-right_bound", m_right_bound, parameters);
			}

			template<typename Scalar>
			void add_scalar_parameter(const std::string &parameter_name, Scalar &variable_value, const Parameters &parameters) {
				
				if (!does_parameter_exist(parameter_name, parameters)) return;
				const std::string parameter_value = parameters.at(parameter_name);

				if (parameter_value != "default")
					variable_value = static_cast<Scalar>(std::stod(parameter_value.c_str()));

				std::cout << parameter_name << " : " << variable_value << std::endl;
			}

			bool does_parameter_exist(const std::string &parameter_name, const Parameters &parameters) {
				
				for (typename Parameters::const_iterator param = parameters.begin(); param != parameters.end(); ++param)
					if ((*param).first == parameter_name) return true;

				return false;
			}

			void clear() {
				
				m_x_data.clear();
				
				m_cmp_roofs_y_data.clear();
				m_cmp_walls_y_data.clear();
				m_cmp_total_y_data.clear();

				m_dst_roofs_y_data.clear();
				m_dst_walls_y_data.clear();

				m_cov_roofs_y_data.clear();
				m_cov_walls_y_data.clear();
				m_cov_total_y_data.clear();
			}

			void compute_x_data(const FT initial_x) {
				
				CGAL_precondition(initial_x > FT(0));
				CGAL_precondition(m_x_data.empty());

				compute_left_x_data(initial_x);
				compute_right_x_data(initial_x);
			}

			void compute_left_x_data(const FT initial_x) {
				
				CGAL_precondition(initial_x > m_left_bound);
				CGAL_precondition(m_left_iters > 0);

				const FT left_length = initial_x - m_left_bound;
				const FT h = left_length / static_cast<FT>(m_left_iters);

				FT count = static_cast<FT>(m_left_iters);
				for (size_t i = 0; i < m_left_iters; ++i, count -= FT(1)) m_x_data.push_back(initial_x - h * count);
				m_x_data.push_back(initial_x);
			}

			void compute_right_x_data(const FT initial_x) {

				CGAL_precondition(m_right_bound > initial_x);
				CGAL_precondition(m_right_iters > 0);

				const FT right_length = m_right_bound - initial_x;
				const FT h = right_length / static_cast<FT>(m_right_iters);

				FT count = FT(1);
				for (size_t i = 0; i < m_right_iters; ++i, count += FT(1)) m_x_data.push_back(initial_x + h * count);
			}

			void compute_y_data() {

				CGAL_precondition(!m_x_data.empty());
				for (size_t i = 0; i < m_x_data.size(); ++i) {
					compute_lod_data(m_x_data[i]);				
					
					set_cpm_y_data();
					set_dst_y_data();
					set_cov_y_data();
				}
			}

			void compute_lod_data(const FT scale) {
				
				m_lod_base.estimate_parameters(false);
				m_lod_base.set_scale(scale);
				m_lod_base.create_lods();
			}

			// Complexity metric!
			void set_cpm_y_data() {
				m_lod_complexity = m_lod_base.get_lod_complexity_ptr();
				
				set_cpm_roofs_y_data();
				set_cpm_walls_y_data();
				set_cpm_total_y_data();
			}

			void set_cpm_roofs_y_data() {
				
				const FT roofs_complexity = m_lod_complexity->get_for_roofs();
				m_cmp_roofs_y_data.push_back(roofs_complexity);
			}

			void set_cpm_walls_y_data() {
				
				const FT walls_complexity = m_lod_complexity->get_for_walls();
				m_cmp_walls_y_data.push_back(walls_complexity);
			}

			void set_cpm_total_y_data() {
				
				const FT total_complexity = m_lod_complexity->get();
				m_cmp_total_y_data.push_back(total_complexity);
			}

			// Distortion metric!
			void set_dst_y_data() {
				m_lod_distortion = m_lod_base.get_lod_distortion_ptr();
				
				set_dst_roofs_y_data();
				set_dst_walls_y_data();
			}

			void set_dst_roofs_y_data() {
				
				const Data &roofs_data = m_lod_distortion->get_roofs_metrics();
				m_dst_roofs_y_data.push_back(roofs_data);
			}

			void set_dst_walls_y_data() {
				
				const Data &walls_data = m_lod_distortion->get_walls_metrics();
				m_dst_walls_y_data.push_back(walls_data);
			}

			// Coverage metric.
			void set_cov_y_data() {
				m_lod_coverage = m_lod_base.get_lod_coverage_ptr();

				set_cov_roofs_y_data();
				set_cov_walls_y_data();
				set_cov_total_y_data();
			}

			void set_cov_roofs_y_data() {
				
				const FT roofs_coverage = m_lod_coverage->get_for_roofs();
				m_cov_roofs_y_data.push_back(roofs_coverage);
			}

			void set_cov_walls_y_data() {
				
				const FT walls_coverage = m_lod_coverage->get_for_walls();
				m_cov_walls_y_data.push_back(walls_coverage);
			}

			void set_cov_total_y_data() {
				
				const FT total_coverage = m_lod_coverage->get();
				m_cov_total_y_data.push_back(total_coverage);
			} */
		};
	
	} // Level_of_detail

} // CGAL

#endif // CGAL_LEVEL_OF_DETAIL_MYQUALITY_ESTIMATOR_H