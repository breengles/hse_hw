#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    vector<int> p(n);
    for (int i = 1; i < n + 1; i++){
        cin >> p[n - i];
    }

    int t;
    cin >> t;
    vector<int> q(n);
    for (int i = 0; i < t; i++){
        int a;
        cin >> a;
        q[n - a] = 1;  // change open/close idx in reversed seq.
    }

    stack<int> v;
    for (int i = 0; i < n; i++){
        if (v.empty()){
            v.push(p[i]);
        }
        else if (p[i] != v.top()){
            // we cant use this kind of brackets to form open-closed subseq. 
            v.push(p[i]);
        }
        else{  // ghosty !v.empty && p[i] == v.top()
            // possible correct subseq. but have to check either or not there must be open bracket
            if (q[i] == 1) {
                // if it must be some open bracket
                v.push(p[i]);
            }
            else{
                // else we can use it to form open-close subseq.
                v.pop();
                p[i] = -p[i];
            }
        }
    }
    if (!v.empty()){
        cout << "NO";
    }
    else{
        cout << "YES" << "\n";
        for (int i = n-1; i >= 0; i--){
            cout << -p[i] << " ";
        }
    }
    return 0;
}