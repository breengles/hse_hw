#include <bits/stdc++.h>

using namespace std;
#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main() {
     unsigned int k, n;

    cin >> n >> k;
    vector<long int> v(n);

    for (unsigned int i = 0; i < n; i++){
        cin >> v[i];
    }

    unsigned long int l = 1;
    unsigned long int r = *max_element(v.begin(), v.end());
    unsigned long int x;

    while (l + 1 < r){
        unsigned long int c = 0;
        
        x = (l + r)/2;
        for (unsigned int i = 0; i < n; i++){
            c += v[i] / x;
        }

        if (c < k){
            r = x - 1;
        }
        else {
            l = x;
        }
    }

    unsigned long int left = 0, right = 0;
    
    for (unsigned int i = 0; i < n; i++){
        left += v[i] / l;
        right += v[i] / r;
    }

    if (right >= k){
        cout << r;
        return 0;
    }
    else if (left >= k){
        cout << l;
        return 0;
    }
    else{
        cout << 0;
    }
    
    return 0;
}