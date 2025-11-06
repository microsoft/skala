/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once

#include <memory>
#include <fstream>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

// Misc string functions

/**
 *  Trim a string of left trailing whitespace
 *
 *  \param [in/out] s std::string to be trimmed
 */
static inline std::string& trim_left(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            [](int ch) { return !std::isspace(ch); }));
    return s;
}; // trim_left


/**
 *  Trim a string of right trailing whitespace
 *
 *  \param [in/out] s std::string to be trimmed
 */
static inline std::string& trim_right(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            [](int ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}; // trim_right


/**
 *  Trim a string of trailing whitespace from both ends
 *
 *  \param [in/out] s std::string to be trimmed
 */
static inline std::string &trim(std::string &s) {
    return trim_left(trim_right(s));
}; // trim

/**
 *  Splits a string into tokens  based on a demiliter
 *
 *  \param [out] tokens     std::vector of std::string objects which hold
 *                          the split tokens
 *  \param [in]  str        std::string to split
 *  \param [in]  delimiters Delimiters on which to split str
 */
static inline void split(std::vector<std::string>& tokens, 
  const std::string& str, const std::string& delimiters = " ") {

    tokens.clear();
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos) {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}; // split

/// INI File Handler 
class INIFile {

  std::shared_ptr<std::ifstream> inFile_ = nullptr;  ///< INI file

  std::unordered_map<std::string,
    std::unordered_map<std::string,std::string>> dict_; 
  ///< INI data fields partitioned by section headings 





  /// Parses the input file
  void parse();

  /// Splits query string on "."
  static std::pair<std::string,std::string> splitQuery(const std::string&);

  /**
   *  std::ofstream constructor.
   *
   *  Sets and parses input file from std::ofstream object
   *  \param [in] inFile  File object to parse
   */  
  INIFile(std::shared_ptr<std::ifstream> inFile) :
    inFile_(inFile){ parse(); }



public:

  // Disable default, copy and move constructors and assignment operators
  INIFile()                           = delete;
  INIFile(const INIFile &)            = delete;
  INIFile(INIFile &&)                 = delete;
  INIFile& operator=(const INIFile &) = delete; 
  INIFile& operator=(INIFile &&)      = delete; 

  /**
   *  Filename constructor.
   *
   *  Sets and parses  input file given a file name
   *  \param [in] inFileName  Name of  input file
   */ 
  INIFile(std::string inFileName) :
    INIFile(std::make_shared<std::ifstream>(inFileName)){ }





  /**
   *  \brief Template function which returns the value of a data field
   *  from the input file in a specified datatype given a formatted 
   *  query string.
   *
   *  i.e.
   *
   *  INI entry:
   *    [SCF]
   *    DENTOL = 1E-6
   *    
   *  Query
   *    double tol = input.getData<double>("SCF.DENTOL");
   *
   *  This example returns the value of the string data field "SCF.DENTOL"
   *  as a double precision number. Various specializations of this function
   *  exist for various datatypes
   *
   *  \param [in] s Formatted query string to be parsed
   *  \return       Value of query data field as specified datatype
   */ 
  template <typename T> T getData(std::string s) ; 




  /**
   *  Checks whether or not the parsed  input file contains
   *  a query section.
   *
   *  \paral  [in] str Query string of a section heading
   *  \return      True if input file contains that heading
   */ 
  inline bool containsSection(std::string str) const {
    return dict_.find(str) != dict_.end();
  }

  /**
   *  Checks whether or not the parsed  input file contains
   *  a query data field.
   *
   *  \paral  [in] str Query string of a data field (includes section heading)
   *  \return      True if input file contains that data field
   */ 
  inline bool containsData(std::string str) const {
    auto pr = splitQuery(str);
    if( not containsSection(pr.first) ) return false;
    return dict_.at(pr.first).find(pr.second) != dict_.at(pr.first).end();
  }





  inline std::vector<std::string> getDataInSection( std::string section )  {

    std::vector<std::string> datasets;

    if( containsSection(section) ) {

      for(auto & data : dict_[section])
        datasets.emplace_back(data.first);

    }

    return datasets;

  }

}; // INIFile class



