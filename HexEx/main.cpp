//#ifndef HEXEXTR_H
//#define HEXEXTR_H
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include"hexextr.h" //checking?
#include<iostream>
#include<string>
//#include<CGAL/Aff_transformation_3.h>
#include"sanitization.h"
#include"typedefs.h"
//namespace HexEx
//#endif



void print_aff_transformation(Aff_transformation T){
  for(int i=0; i<4; i++){
    for(int j = 0; j<4; j++)
      std::cout<<T.m(i,j)<<" ";
    std::cout<<std::endl; 
    }
  std::cout<<std::endl;
  return;
}

int main(int argc, char** argv){
  std::string str;
  if (argc==1)
  {
    std::cout<<"Enter filename"<<std::endl;
    std::cin>>str;
  }
  else
  {
    str=argv[1];
  }
  HexExtr h(str);
  //for(int i = 0; i<24;i++) print_aff_tranformation(h.G[i]);

}

