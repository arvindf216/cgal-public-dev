#ifndef GENERATOR_H
#define GENERATOR_H

#include "types.h"
#include "qem.h"

namespace qem
{
    template <class Kernel>
    class CGenerator
    {
    private:
        int m_point_index; //  index in input point set
        Point m_location; // optimal location in space
        QEM_metric m_qem; // sum of qem of the whole cluster
        bool m_is_active; // only for ftc, if a generator is active currently

    public:
        int& point_index() { return m_point_index; }
        const int& point_index() const { return m_point_index; }

        Point& location() { return m_location; }
        const Point& location() const { return m_location; }

        QEM_metric& qem() { return m_qem; }
        const QEM_metric& qem() const { return m_qem; }

        bool& is_active() { return m_is_active; }
        const bool& is_active() const { return m_is_active; }

    public:
        // default constructor
        CGenerator()
        {
            m_point_index = 0;
            m_qem = QEM_metric();
            m_location = CGAL::ORIGIN;
            m_is_active = true;
        }

        // constructor with null qem
        CGenerator(const int point_index,
            const Point& location)
        {
            m_point_index = point_index;
            m_location = location;
            m_is_active = true;
        }

        // copy constructor
        CGenerator(const CGenerator& g)
            : m_point_index(g.point_index()),
            m_location(g.location()), 
            m_qem(g.qem()),
            m_is_active(g.is_active())
        {
        }

        // add qem matrix
        void add_qem(const QEM_metric& qem)
        {
            m_qem = m_qem + qem;
        }
    };
}

#endif /* GENERATOR_H */
