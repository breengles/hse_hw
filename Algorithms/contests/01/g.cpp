#include <bits/stdc++.h>

using namespace std;

#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    stack<long> v;

    string s;
    getline(cin, s);
    stringstream ss(s);

    string tmp;
    long x;
    char c;
    while (!ss.eof()){
        ss >> tmp;
        if (stringstream(tmp) >> x){
            v.push(x);
        }
        else{
            long y;
            stringstream(tmp) >> c;
            x = v.top();
            v.pop();
            y = v.top();
            v.pop();
            if (c == '+'){
                v.push(y + x);
            }
            else if (c == '*'){
                v.push(y*x);
            }
            else{
                v.push(y - x);
            }
        }
    }
    cout << v.top() << "\n";
    return 0;
}