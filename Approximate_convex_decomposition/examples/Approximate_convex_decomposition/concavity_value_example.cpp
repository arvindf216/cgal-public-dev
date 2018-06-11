#include <CGAL/approx_decomposition.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>

#include <iostream>
#include <fstream>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Polyhedron_3<Kernel>     Polyhedron;

int main()
{
    // read mesh
    Polyhedron mesh;
    
    std::ifstream input("data/elephant.off");
    
    if (!input || !(input >> mesh))
    {
        std::cout << "Failed to read mesh" << std::endl;
        return EXIT_FAILURE;
    }

    if (CGAL::is_empty(mesh) || !CGAL::is_triangle_mesh(mesh))
    {
        std::cout << "Input mesh is invalid" << std::endl;
        return EXIT_FAILURE;
    }

    // compute concavity value
//    double concavity = CGAL::concavity_value(mesh);

    // write result
//    std::cout << "Concavity value: " << concavity << std::endl;

    return EXIT_SUCCESS;
}
