//
// Created by elfuius on 05/03/18.
//

#include "schlepilUtils.h"

void checkEndOfPath(string & path){
    if (not(path.back() == '/')){
        path += "/";
    }
}