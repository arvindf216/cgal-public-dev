#ifndef CGAL_LEVEL_OF_DETAIL_PARTITION_ELEMENT_H
#define CGAL_LEVEL_OF_DETAIL_PARTITION_ELEMENT_H

// LOD includes.
#include <CGAL/Level_of_detail/Enumerations.h>

namespace CGAL {

namespace Level_of_detail {

template<class GeomTraits, class Container>
class Partition_element {
        
public:
  using Kernel    = GeomTraits;

  using FT = typename Kernel::FT;
  using const_iterator = typename Container::Vertex_const_iterator;

  template<class Elements, class Point_map>
  Partition_element(const Elements &elements, const Point_map &point_map) {
    m_visibility_label = Visibility_label::OUTSIDE;
    m_container.clear();
    using Const_elements_iterator = typename Elements::const_iterator;
                
    for (Const_elements_iterator ce_it = elements.begin(); ce_it != elements.end(); ++ce_it)
      m_container.push_back(get(point_map, *ce_it));
  }

  Partition_element()
  {
  }

  void push_back (const typename Container::Point_2& point)
  {
    m_container.push_back(point);
  }

  template<class Point>
  inline bool has_on_bounded_side(const Point &query) const {
    if (!m_container.is_simple())
      return false;
    return m_container.has_on_bounded_side(query);
  }

  inline const const_iterator begin() const {
    return m_container.vertices_begin();
  }

  inline const const_iterator end() const {
    return m_container.vertices_end();
  }

  inline Visibility_label& visibility_label() {
    return m_visibility_label;
  }

  inline const Visibility_label& visibility_label() const {
    return m_visibility_label;
  }

  inline size_t size() const {
    return m_container.size();
  }

  inline const typename Container::Point_2& operator[] (std::size_t idx) const {
    return m_container[idx];
  }

private:
  Container              m_container;
  Visibility_label       m_visibility_label;
};

} // Level_of_detail

} // CGAL

#endif // CGAL_LEVEL_OF_DETAIL_PARTITION_ELEMENT_H